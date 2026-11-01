"""Run the CASPER-U main model with leave-one-speaker-out evaluation."""

import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score, accuracy_score, f1_score
import torch
import torch.nn.functional as F
from .config import parse_args, TARGET_CLASSES, UNIFIED
from .model import SharedHeads, _GradReverse
from .utils import set_seed, speaker_center, group_assignment_probs


def load_all(datasets, layers, manifest_dir, feature_dir):
    """Align each manifest to its feature cache using unique utterance IDs."""
    out = {}
    for ds in datasets:
        df = pd.read_csv(os.path.join(manifest_dir, f"{ds}.csv"), dtype={"utt_id": str})
        required = {"utt_id", "label", "speaker", "sentence", "dialogue"}
        if not required.issubset(df.columns):
            raise ValueError(f"{ds}: required manifest columns: {sorted(required)}")
        if df[list(required)].isna().any().any() or df.utt_id.duplicated().any():
            raise ValueError(
                f"{ds}: metadata must be nonempty and utterance IDs unique"
            )
        with np.load(
            os.path.join(feature_dir, f"{ds}_whisperL.npz"), allow_pickle=False
        ) as z:
            ids = z["utt_ids"].astype(str)
            if len(set(ids)) != len(ids) or set(df.utt_id) != set(ids):
                raise ValueError(f"{ds}: feature/manifest utterance IDs do not match")
            df = df.set_index("utt_id").loc[list(ids)].reset_index()
            classes = TARGET_CLASSES[ds]
            keep = df.label.isin(classes).values
            mean, std = (z["mean"], z["std"])
            if (
                mean.shape != std.shape
                or mean.ndim != 3
                or mean.shape[0] != len(df)
                or (max(layers) >= mean.shape[1])
            ):
                raise ValueError(f"{ds}: invalid per-layer feature shapes")
            blocks = [
                np.concatenate(
                    [mean[:, l, :].astype(np.float32), std[:, l, :].astype(np.float32)],
                    1,
                )[keep]
                for l in layers
            ]
        df = df[keep].reset_index(drop=True)
        if df.empty or not all((np.isfinite(b).all() for b in blocks)):
            raise ValueError(f"{ds}: empty inventory or non-finite features")
        df["y_local"] = df.label.map({c: i for i, c in enumerate(classes)})
        df["y_uni"] = df.label.map({c: i for i, c in enumerate(UNIFIED)})
        df["speaker_uid"] = ds + "::" + df.speaker.astype(str)
        out[ds] = dict(df=df, blocks=blocks, classes=classes)
        print(f"{ds}: {len(df)} utterances, {len(classes)} classes")
    return out


def main():
    args = parse_args()
    set_seed(args.seed)
    datasets = args.datasets.split(",")
    layers = [int(x) for x in args.layers.split(",")]
    data = load_all(datasets, layers, args.manifest_dir, args.feature_dir)
    target = args.target
    tdf = data[target]["df"]
    target_classes = data[target]["classes"]
    ds_id = {ds: i for i, ds in enumerate(datasets)}
    for ds in datasets:
        data[ds]["ds_idx"] = np.full(len(data[ds]["df"]), ds_id[ds], np.int64)
        blocks = data[ds]["blocks"]
        data[ds]["blocks_c"] = speaker_center(
            blocks, data[ds]["df"].speaker_uid.values, False
        )
    speakers = tdf.speaker_uid.values
    uniq = sorted(set(speakers))
    test_list = uniq
    class_counts = (
        [len(UNIFIED)]
        if args.single_head
        else [len(data[ds]["classes"]) for ds in datasets]
    )
    target_yidx = [UNIFIED.index(c) for c in target_classes]
    results = []
    all_pred = {}
    t0 = time.time()
    for g in test_list:
        te_mask = speakers == g
        n_target_train = int((~te_mask).sum())
        n_blocks = len(data[target]["blocks"])
        Xtr_layers = [[] for _ in range(n_blocks)]
        Xte_layers = [[] for _ in range(n_blocks)]
        ytr_parts, wtr_parts, dtr_parts, lengths = ([], [], [], [])
        for ds in datasets:
            d = data[ds]
            if ds == target:
                tr = ~te_mask
                te = te_mask
            else:
                tr = np.ones(len(d["df"]), bool)
                te = None
            for li in range(n_blocks):
                Xtr_layers[li].append(d["blocks_c"][li][tr])
                if te is not None:
                    Xte_layers[li].append(d["blocks_c"][li][te])
            ylab = (d["df"].y_uni if args.single_head else d["df"].y_local).values
            ytr_parts.append(ylab[tr])
            dtr_parts.append(d["ds_idx"][tr])
            n_tr = int(tr.sum())
            lengths.append(n_tr)
            if ds == target:
                w = np.full(n_tr, args.target_weight, np.float32)
                wtr_parts.append(w.astype(np.float32))
            else:
                w = args.ext_weight * n_target_train / max(n_tr, 1)
                wtr_parts.append(np.full(n_tr, w, np.float32))
        Ztr_parts, Zte_parts = ([], [])
        for li in range(n_blocks):
            Btr = np.concatenate(Xtr_layers[li], 0)
            Bte = np.concatenate(Xte_layers[li], 0)
            sc = StandardScaler().fit(Btr)
            Btr = sc.transform(Btr)
            Bte = sc.transform(Bte)
            k = min(args.per_layer_pca, Btr.shape[0] - 1, Btr.shape[1])
            pc = PCA(
                n_components=k, svd_solver="randomized", random_state=args.seed
            ).fit(Btr)
            Ztr_parts.append(pc.transform(Btr))
            Zte_parts.append(pc.transform(Bte))
        Ztr = np.concatenate(Ztr_parts, 1)
        Zte = np.concatenate(Zte_parts, 1)
        Ztr_parts = np.split(Ztr, np.cumsum(lengths)[:-1], axis=0)
        extra_dim = 0
        if args.iemo_context:
            extra_dim += args.iemo_context_pca
        if args.iemo_prosody:
            extra_dim += args.iemo_prosody_pca
        if extra_dim > 0:
            Ztr_parts = [
                np.concatenate([p, np.zeros((len(p), extra_dim), np.float32)], 1)
                for p in Ztr_parts
            ]
            Zte = np.concatenate([Zte, np.zeros((len(Zte), extra_dim), np.float32)], 1)
            if target == "IEMOCAP":
                ti = ds_id[target]
                off = 0
                if args.iemo_context:
                    D = Ztr_parts[ti].shape[1]
                    Zall = np.zeros((len(tdf), D), np.float32)
                    Zall[~te_mask] = Ztr_parts[ti][:, :D]
                    Zall[te_mask] = Zte[:, :D]
                    C_all = np.zeros((len(tdf), 3 * D), np.float32)
                    dial = tdf.dialogue.values
                    spk_all = tdf.speaker.values
                    for d in np.unique(dial):
                        idx = np.where(dial == d)[0]
                        for k in idx:
                            oth = idx[idx != k]
                            if len(oth) == 0:
                                continue
                            C_all[k, :D] = Zall[oth].mean(0)
                            same = oth[spk_all[oth] == spk_all[k]]
                            other = oth[spk_all[oth] != spk_all[k]]
                            if len(same):
                                C_all[k, D : 2 * D] = Zall[same].mean(0)
                            if len(other):
                                C_all[k, 2 * D :] = Zall[other].mean(0)
                    Ctr = C_all[~te_mask]
                    Cte = C_all[te_mask]
                    cs = StandardScaler().fit(Ctr)
                    Ctr = cs.transform(Ctr)
                    Cte = cs.transform(Cte)
                    kc = min(args.iemo_context_pca, Ctr.shape[0] - 1, Ctr.shape[1])
                    cp = PCA(
                        n_components=kc, svd_solver="randomized", random_state=args.seed
                    ).fit(Ctr)
                    Ztr_parts[ti][:, off : off + kc] = cp.transform(Ctr)
                    Zte[:, off : off + kc] = cp.transform(Cte)
                    off += kc
                if args.iemo_prosody:
                    pz = np.load(
                        os.path.join(args.feature_dir, "IEMOCAP_prosody.npz"),
                        allow_pickle=True,
                    )
                    pm = {u: i for i, u in enumerate(pz["utt_ids"])}
                    P = pz["X"][[pm[u] for u in tdf.utt_id.values]].astype(np.float32)
                    Ptr, Pte = (P[~te_mask], P[te_mask])
                    ps = StandardScaler().fit(Ptr)
                    Ptr = ps.transform(Ptr)
                    Pte = ps.transform(Pte)
                    kp = min(args.iemo_prosody_pca, Ptr.shape[0] - 1, Ptr.shape[1])
                    pp = PCA(
                        n_components=kp, svd_solver="randomized", random_state=args.seed
                    ).fit(Ptr)
                    Ztr_parts[ti][:, off : off + kp] = pp.transform(Ptr)
                    Zte[:, off : off + kp] = pp.transform(Pte)
        Ztr = np.concatenate(Ztr_parts, 0)
        ytr = np.concatenate(ytr_parts)
        dtr = np.concatenate(dtr_parts)
        wtr = np.concatenate(wtr_parts)
        spk_parts = []
        for ds in datasets:
            d = data[ds]
            if ds == target:
                spk_parts.append(d["df"].speaker_uid.values[~te_mask])
            else:
                spk_parts.append(d["df"].speaker_uid.values)
        spk_all = np.concatenate(spk_parts)
        spk_uniq = sorted(set(spk_all))
        spk_map = {s: i for i, s in enumerate(spk_uniq)}
        spk_ids = np.array([spk_map[s] for s in spk_all], np.int64)
        if args.single_head:
            n_cls_cw = len(UNIFIED)
            if args.no_class_weights:
                cw_list = [None]
            else:
                cnt = np.bincount(ytr, minlength=n_cls_cw).astype(np.float32)
                cw_list = [
                    torch.tensor(
                        cnt.sum() / (n_cls_cw * np.maximum(cnt, 1)), dtype=torch.float32
                    )
                ]
        else:
            cw_list = []
            for yi, cc in zip(ytr_parts, class_counts):
                if args.no_class_weights:
                    cw_list.append(None)
                else:
                    cnt = np.bincount(yi, minlength=cc).astype(np.float32)
                    cw_list.append(
                        torch.tensor(
                            cnt.sum() / (cc * np.maximum(cnt, 1)), dtype=torch.float32
                        )
                    )
        model = SharedHeads(
            Ztr.shape[1], class_counts, args.hidden, args.dropout, len(spk_uniq)
        )
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        ytr_t = torch.tensor(ytr, dtype=torch.long)
        dtr_t = torch.tensor(dtr, dtype=torch.long)
        wtr_t = torch.tensor(wtr, dtype=torch.float32)
        Xtr_t = torch.tensor(Ztr, dtype=torch.float32)
        Xte_t = torch.tensor(Zte, dtype=torch.float32)
        n = len(Xtr_t)
        probs = []
        for ep in range(args.epochs):
            model.train()
            perm = torch.randperm(n)
            for i in range(0, n, args.bs):
                idx = perm[i : i + args.bs]
                yb, db, wb = (ytr_t[idx], dtr_t[idx], wtr_t[idx])
                xb = Xtr_t[idx]
                h = model.encode(xb, db)
                if args.single_head:
                    logits = model.heads[0](h)
                    loss = (
                        F.cross_entropy(
                            logits,
                            yb,
                            weight=cw_list[0],
                            reduction="none",
                            label_smoothing=args.ls,
                        )
                        * wb
                    ).sum() / wb.sum().clamp_min(1e-06)
                else:
                    loss = torch.tensor(0.0)
                    for dd in torch.unique(db):
                        m = db == dd
                        logits = model.heads[int(dd)](h[m])
                        loss = (
                            loss
                            + (
                                F.cross_entropy(
                                    logits,
                                    yb[m],
                                    weight=cw_list[int(dd)],
                                    reduction="none",
                                    label_smoothing=args.ls,
                                )
                                * wb[m]
                            ).sum()
                        )
                    loss = loss / wb.sum().clamp_min(1e-06)
                if args.adv_weight > 0:
                    slog = model.spk(_GradReverse.apply(h, args.adv_weight))
                    loss = loss + args.adv_weight * F.cross_entropy(
                        slog, torch.tensor(spk_ids[idx], dtype=torch.long)
                    )
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
            if ep >= args.epochs - args.avg_last:
                model.eval()
                with torch.no_grad():
                    tds = torch.full((len(Xte_t),), ds_id[target], dtype=torch.long)
                    logits = model(
                        Xte_t, tds, target_head=0 if args.single_head else ds_id[target]
                    )
                    probs.append(F.softmax(logits, -1).numpy())
        p = np.mean(probs, 0)
        if args.single_head:
            p = p[:, target_yidx]
            p = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
        if args.pspc and target in ("CASIA", "BodEMODB"):
            cost = np.log(p + 1e-12)
            sent_eval = tdf.sentence.values[te_mask].copy()
            p_perm = group_assignment_probs(
                cost, sent_eval, tdf.speaker.values[te_mask], len(target_classes)
            )
            p = (1 - args.pspc_lambda) * p + args.pspc_lambda * p_perm
        yte = tdf.y_local.values[te_mask]
        pred = p.argmax(1)
        m = dict(
            fold=str(g),
            n=int(te_mask.sum()),
            uar=float(recall_score(yte, pred, average="macro", zero_division=0)),
            war=float(accuracy_score(yte, pred)),
            f1=float(f1_score(yte, pred, average="macro", zero_division=0)),
        )
        results.append(m)
        all_pred[str(g)] = dict(
            y=[int(v) for v in yte],
            p=p.tolist(),
            utt=tdf.utt_id.values[te_mask].tolist(),
        )
        print(f"{g}: UAR {m['uar']:.4f} WAR {m['war']:.4f}")
    uar = np.array([r["uar"] for r in results])
    war = np.array([r["war"] for r in results])
    print(
        json.dumps(
            dict(
                target=target,
                n_folds=len(results),
                uar_mean=float(uar.mean()),
                uar_std=float(uar.std()),
                war_mean=float(war.mean()),
                runtime_s=round(time.time() - t0, 1),
            ),
            indent=2,
        )
    )
    np.save(
        os.path.join(args.output_dir, f"casper_u_{target}_{args.tag}_preds.npy"),
        all_pred,
        allow_pickle=True,
    )
    with open(
        os.path.join(args.output_dir, f"casper_u_{target}_{args.tag}.json"), "w"
    ) as f:
        json.dump(
            dict(
                vars(args),
                per_fold=results,
                uar_mean=float(uar.mean()),
                uar_std=float(uar.std()),
                war_mean=float(war.mean()),
            ),
            f,
            indent=2,
            default=str,
        )


if __name__ == "__main__":
    main()
