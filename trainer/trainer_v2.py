import argparse
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TASKS = {
    "A": {"dir": "cpu-a", "kind": "classification",
          "classes": ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"],
          "metric": "macro_f1", "higher_better": True},
    "B": {"dir": "cpu-b", "kind": "regression", "classes": [],
          "metric": "rmse", "higher_better": False},
    "C": {"dir": "cpu-c", "kind": "regression", "classes": [],
          "metric": "rmse", "higher_better": False},
}

def macro_f1(y_true, y_pred, n_classes):
    f1s = []
    for c in range(n_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        if tp + fp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0

def per_class_f1(y_true, y_pred, classes):
    out = {}
    for i, name in enumerate(classes):
        tp = int(np.sum((y_pred == i) & (y_true == i)))
        fp = int(np.sum((y_pred == i) & (y_true != i)))
        fn = int(np.sum((y_pred != i) & (y_true == i)))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out[name] = round(2 * p * r / (p + r) if p + r else 0.0, 4)
    return out

def rmse_mae(y_true, y_pred):
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sqrt(np.mean(d ** 2))), float(np.mean(np.abs(d)))

def nasa_score(y_true, y_pred):
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    s = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(s))

def psi(ref, cur, bins=10):
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    qs = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if qs.size < 3:
        return 0.0
    qs[0], qs[-1] = -np.inf, np.inf
    r_hist, _ = np.histogram(ref, bins=qs)
    c_hist, _ = np.histogram(cur, bins=qs)
    r_pct = np.clip(r_hist / max(len(ref), 1), 1e-6, None)
    c_pct = np.clip(c_hist / max(len(cur), 1), 1e-6, None)
    return float(np.sum((c_pct - r_pct) * np.log(c_pct / r_pct)))

def bootstrap_gain(y_true, pred_before, pred_after, kind, n_classes,
                   higher_better, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    gains = np.empty(n_boot)
    if kind == "classification":
        yb_before = np.argmax(pred_before, axis=1)
        yb_after = np.argmax(pred_after, axis=1)
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            yt = y_true[idx]
            g = macro_f1(yt, yb_after[idx], n_classes) - macro_f1(yt, yb_before[idx], n_classes)
            gains[b] = g
    else:
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            d0 = pred_before[idx] - y_true[idx]
            d1 = pred_after[idx] - y_true[idx]
            gains[b] = float(np.sqrt(np.mean(d0 ** 2)) - np.sqrt(np.mean(d1 ** 2)))
    if not higher_better:
        pass
    return gains

def ci_summary(gains, alpha=0.05):
    return {
        "gain_mean": float(np.mean(gains)),
        "gain_p5": float(np.percentile(gains, alpha * 100)),
        "gain_p50": float(np.percentile(gains, 50)),
        "gain_p95": float(np.percentile(gains, (1 - alpha) * 100)),
        "p_positive": float(np.mean(gains > 0)),
        "n_boot": int(len(gains)),
    }

def relative_drop(before, after, higher_better):
    if before == 0:
        return 0.0
    if higher_better:
        return float((before - after) / abs(before))
    return float((after - before) / abs(before))

def policy_from_drift(psi_mean, psi_max, requested_epochs, requested_lr):
    if psi_mean < 0.10:
        action, epochs, lr, over = "hold_no_retrain", 0, None, 1
        reason = "漂移可忽略（PSI 均值 < 0.10），不重训，省一轮"
    elif psi_mean < 0.25:
        action, epochs, lr, over = "small_step", 3, 0.001, 1
        reason = "轻度漂移（0.10~0.25），小步续训 + 低学习率，避免过冲"
    else:
        action, epochs, lr, over = "big_step", 12, 0.003, 1
        reason = "显著漂移（> 0.25），大步续训（默认不重加权，实测过采样对 B 有害）"
    if psi_max >= 1.0 and action == "small_step":
        action, epochs, lr, over = "big_step", 12, 0.003, 1
        reason = "单维漂移极显著（PSI 最大 ≥ 1.0），升格为大步"
    if requested_epochs is not None:
        epochs = requested_epochs
        if action == "hold_no_retrain":
            action = "manual"
    if requested_lr is not None:
        lr = requested_lr
    return {"psi_mean": round(float(psi_mean), 4), "psi_max": round(float(psi_max), 4),
            "action": action, "epochs": int(epochs), "lr": lr,
            "oversample_new": int(over), "reason": reason}

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["A", "B", "C"])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--from-baseline", nargs="?", const="official", default="official")
    ap.add_argument("--from-last", choices=["latest", "best"], default=None)
    ap.add_argument("--epochs", type=int, default=None,
                    help="不传则由策略层按漂移强度决定（--policy off 时用配置文件的值）")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--iter", type=int, default=1)
    ap.add_argument("--mode", default="local", choices=["local", "online", "offline", "disabled"])
    ap.add_argument("--project", default="rsi-harness")
    ap.add_argument("--workspace", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--logdir", default=str(HERE.parent / "swanlog"))
    ap.add_argument("--no-swan", action="store_true")
    ap.add_argument("--val-data", default=None)
    ap.add_argument("--extra-data", action="append", default=[])
    ap.add_argument("--ood-data", default=None,
                    help="OOD 验证集（通常是上一批的留出集）；给了就启用 M1 的 OOD 闸门")
    ap.add_argument("--promote-delta", type=float, default=None,
                    help="兼容 v1 的点估计阈值；只在 --gate point 时生效")
    ap.add_argument("--policy", default="auto", choices=["auto", "off"])
    ap.add_argument("--oversample", type=int, default=None,
                    help="新批次过采样倍数；默认跟随策略（当前为 1，即不重加权）")
    ap.add_argument("--patience", type=int, default=5, help="0 = 关闭早停（等价 v1）")
    ap.add_argument("--bootstrap", type=int, default=500, help="bootstrap 次数，0 = 关闭")
    ap.add_argument("--gate", default="ci", choices=["ci", "point"],
                    help="ci = 区间下界 > 0 才放行；point = v1 的点估计阈值")
    ap.add_argument("--ood-tolerance", type=float, default=0.03)
    ap.add_argument("--hold-action", default="partial",
                    choices=["partial", "skip"], help="策略判定不重训时：partial=仍跑 1 轮记录")
    args = ap.parse_args(argv)

    spec = TASKS[args.task]
    pkg = (Path(args.data_dir) if args.data_dir else HERE / spec["dir"]).resolve()
    if not (pkg / "baselines" / "data.py").is_file():
        raise SystemExit(f"找不到任务包: {pkg}")
    sys.path.insert(0, str(pkg))
    from baselines.data import read_dataset
    from baselines.model import Model, checkpoint_digest

    cfg = {}
    cfg_path = pkg / "baselines" / "configs" / f"{args.task}.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    hidden = args.hidden if args.hidden is not None else int(cfg.get("hidden", 32))
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    batch = args.batch if args.batch is not None else int(cfg.get("batch_size", 128))
    classes = spec["classes"]

    t0 = time.time()
    print(f"[1/6] 读数据  task={args.task}  包={pkg.name}", flush=True)
    kwargs = {"classes": classes} if spec["kind"] == "classification" else {}
    base = read_dataset(pkg / "public" / "train.jsonl", spec["kind"], **kwargs)
    tr = base
    n_new = 0
    if args.extra_data:
        xs, ys, ids = [base["x"]], [base["y"]], list(base["ids"])
        for extra in args.extra_data:
            ep = Path(extra)
            if not ep.is_absolute():
                ep = (pkg / ep).resolve()
            ex = read_dataset(ep, spec["kind"], expected_paths=base["paths"], **kwargs)
            xs.append(ex["x"]); ys.append(ex["y"]); ids += list(ex["ids"])
            n_new += len(ex["ids"])
            print(f"      + 并入新批次 {ep.name}：{len(ex['ids'])} 条", flush=True)
        tr = {"x": np.vstack(xs), "y": np.concatenate(ys), "ids": ids, "paths": base["paths"]}

    if args.val_data:
        vp = Path(args.val_data)
        if not vp.is_absolute():
            vp = (pkg / vp).resolve()
        va = read_dataset(vp, spec["kind"], expected_paths=base["paths"], **kwargs)
    else:
        va = read_dataset(pkg / "public" / "validation.jsonl", spec["kind"],
                          expected_paths=base["paths"], **kwargs)
    ood = None
    if args.ood_data:
        op = Path(args.ood_data)
        if not op.is_absolute():
            op = (pkg / op).resolve()
        ood = read_dataset(op, spec["kind"], expected_paths=base["paths"], **kwargs)
        print(f"      OOD 集 {op.name}：{len(ood['ids'])} 条", flush=True)
    print(f"      train {len(tr['ids'])}（新批次 {n_new}）/ val {len(va['ids'])} / 特征 {len(tr['paths'])} 维", flush=True)

    def evaluate(m, ds=None):
        d = ds if ds is not None else va
        pred = m.predict(d["x"])
        if spec["kind"] == "classification":
            yp = np.argmax(pred, axis=1)
            return {"acc": float(np.mean(yp == d["y"])),
                    "macro_f1": macro_f1(d["y"], yp, len(classes)),
                    "per_class_f1": per_class_f1(d["y"], yp, classes)}
        rmse, mae = rmse_mae(d["y"], pred)
        out = {"rmse": rmse, "mae": mae}
        if args.task == "C":
            out["nasa"] = nasa_score(d["y"], pred)
        return out

    def predict_of(m, ds):
        return m.predict(ds["x"])

    print("[2/6] 漂移检测 PSI", flush=True)
    drift_ref = base["x"]
    drift_cur = va["x"] if n_new == 0 else tr["x"][len(base["ids"]):]
    per_psi = np.array([psi(drift_ref[:, i], drift_cur[:, i]) for i in range(base["x"].shape[1])])
    drift = {"psi_mean": float(per_psi.mean()), "psi_max": float(per_psi.max()),
             "psi_frac_gt_0_25": float((per_psi > 0.25).mean()),
             "ref_rows": len(base["ids"]), "cur_rows": int(len(drift_cur))}
    print(f"      均值 {drift['psi_mean']:.3f} / 最大 {drift['psi_max']:.3f} "
          f"/ {drift['psi_frac_gt_0_25']*100:.0f}% 维度 > 0.25", flush=True)

    if args.policy == "auto":
        pol = policy_from_drift(drift["psi_mean"], drift["psi_max"], args.epochs, args.lr)
    else:
        pol = {"psi_mean": round(drift["psi_mean"], 4), "psi_max": round(drift["psi_max"], 4),
               "action": "manual_off", "epochs": int(args.epochs if args.epochs is not None else cfg.get("epochs", 15)),
               "lr": args.lr, "oversample_new": 1, "reason": "policy=off，按固定参数执行"}
    if args.oversample is not None:
        pol["oversample_new"] = int(args.oversample)
    if pol["epochs"] == 0 and args.epochs is None and args.hold_action == "partial" and n_new:
        pol["epochs"] = 1
        pol["reason"] += "（hold 降级为 1 轮记录）"
    epochs = int(pol["epochs"])
    lr = pol["lr"]
    print(f"      策略 {pol['action']}：epochs={epochs} lr={lr} 新批过采样×{pol['oversample_new']}"
          f" —— {pol['reason']}", flush=True)

    if pol["oversample_new"] > 1 and n_new:
        k = pol["oversample_new"]
        tail_x = tr["x"][len(base["ids"]):]
        tail_y = tr["y"][len(base["ids"]):]
        tr = {"x": np.vstack([tr["x"], np.repeat(tail_x, k - 1, axis=0)]),
              "y": np.concatenate([tr["y"], np.repeat(tail_y, k - 1)]),
              "ids": tr["ids"] + [f"{i}#dup{j}" for j in range(k - 1) for i in tr["ids"][len(base["ids"]):]],
              "paths": tr["paths"]}
        print(f"      新批次过采样 ×{k} → 训练集 {len(tr['y'])} 条", flush=True)

    official = pkg / "checkpoint.npz"
    runs_dir = pkg / "runs"
    run_dirs = sorted([p for p in runs_dir.glob(f"task{args.task}-iter*-*")
                       if (p / "checkpoint.npz").is_file()], key=lambda p: p.stat().st_mtime) if runs_dir.is_dir() else []

    print("[3/6] 选父检查点 / 建模型", flush=True)
    if args.from_last == "latest":
        parent_path = (run_dirs[-1] / "checkpoint.npz") if run_dirs else official
        print(f"      from-last latest → {parent_path.parent.name if run_dirs else '官方基线'}", flush=True)
    elif args.from_last == "best":
        cands = [(official, "官方基线")] + [(d / "checkpoint.npz", d.name) for d in run_dirs]
        best_path, best_tag, best_val = official, "官方基线", None
        for path, tag in cands:
            val = evaluate(Model.load(path))[spec["metric"]]
            better = (val > best_val) if (best_val is not None and spec["higher_better"]) else \
                     (val < best_val) if best_val is not None else True
            if best_val is None or better:
                best_val, best_path, best_tag = val, path, tag
        parent_path = best_path
        print(f"      from-last best → {best_tag}（{spec['metric']} {best_val:.4f}）", flush=True)
    else:
        parent_arg = args.from_baseline
        if parent_arg in (None, "", "official"):
            parent_path = official
        elif str(parent_arg).lower() == "none":
            parent_path = None
        else:
            parent_path = Path(parent_arg)
            if not parent_path.is_absolute():
                parent_path = (Path.cwd() / parent_path).resolve()

    parent = None
    if parent_path is not None:
        if not parent_path.is_file():
            raise SystemExit(f"父检查点不存在: {parent_path}")
        model = Model.load(parent_path)
        parent = checkpoint_digest(parent_path)
        if len(model.meta["paths"]) != len(tr["paths"]):
            raise SystemExit(f"特征维度不一致: 检查点 {len(model.meta['paths'])} vs 数据 {len(tr['paths'])}")
        print(f"      续训 {parent_path.parent.name}/{parent_path.name}（父 {parent[:23]}…，已训 {model.meta['epochs']} 轮）", flush=True)
    else:
        model = Model(spec["kind"], tr["paths"], classes, tr["x"], tr["y"],
                      hidden=hidden, seed=seed, learning_rate=lr or float(cfg.get("learning_rate", 0.003)),
                      batch_size=batch)
        print("      从零训练", flush=True)
    if lr is not None:
        model.meta["learning_rate"] = float(lr)
    print(f"      参数量 {model.meta['parameter_count']}", flush=True)

    before = evaluate(model)
    before_pred = predict_of(model, va)
    ood_before = evaluate(model, ood) if ood is not None else None
    print(f"      before: " + json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                        for k, v in before.items() if k != "per_class_f1"}, ensure_ascii=False), flush=True)

    swan = None
    if not args.no_swan:
        try:
            import swanlab
            swanlab.init(project=args.project, workspace=args.workspace,
                         experiment_name=args.run_name or f"task{args.task}-iter{args.iter:02d}",
                         mode=args.mode, log_dir=args.logdir,
                         config={"task": args.task, "hidden": hidden, "seed": seed, "lr": lr,
                                 "batch": batch, "epochs": epochs, "parent": parent or "none",
                                 "parent_tag": str(parent_path.parent.name if parent_path else "scratch"),
                                 "param_count": model.meta["parameter_count"],
                                 "policy": pol["action"], "psi_mean": drift["psi_mean"]})
            swan = swanlab
            print(f"      SwanLab 已连接（{args.mode}）", flush=True)
        except Exception as exc:
            print(f"      SwanLab 不可用，降级为纯本地: {exc}", flush=True)

    print(f"[4/6] 训练  {epochs} 轮（patience={args.patience}）", flush=True)
    history = []
    base_epoch = model.meta["epochs"]
    best_model = copy.deepcopy(model)
    best_metric = before[spec["metric"]]
    best_epoch = base_epoch
    waited = 0
    stopped_at = base_epoch
    for ep in range(1, epochs + 1):
        loss = model.fit(tr["x"], tr["y"], 1)[-1]
        s = evaluate(model)
        v = s[spec["metric"]]
        better = (v > best_metric) if spec["higher_better"] else (v < best_metric)
        if better:
            best_metric, best_epoch, best_model = v, base_epoch + ep, copy.deepcopy(model)
            waited = 0
        else:
            waited += 1
        history.append({"epoch": base_epoch + ep, "loss": float(loss),
                        "is_best": bool(better),
                        **{k: val for k, val in s.items() if k != "per_class_f1"}})
        if swan and (ep % 5 == 0 or ep == epochs or better is False and waited == args.patience):
            log = {"loss": float(loss), "psi_mean": drift["psi_mean"], "psi_max": drift["psi_max"],
                   "is_best": 1 if better else 0}
            if spec["kind"] == "classification":
                log.update({"val_acc": s["acc"], "val_macro_f1": s["macro_f1"]})
                for name, val in s["per_class_f1"].items():
                    log[f"f1_{name}"] = val
            else:
                log.update({"val_rmse": s["rmse"], "val_mae": s["mae"]})
                if "nasa" in s:
                    log["val_nasa"] = s["nasa"]
            swan.log(log, step=base_epoch + ep)
        print(f"      epoch {base_epoch + ep:3d}  loss {loss:8.4f}  {spec['metric']} {v:9.4f}"
              f"{'  *best' if better else ''}", flush=True)
        stopped_at = base_epoch + ep
        if args.patience > 0 and waited >= args.patience:
            print(f"      早停：连续 {args.patience} 轮无提升（best 在第 {best_epoch} 轮）", flush=True)
            break

    if args.patience > 0 and epochs > 0:
        model = best_model
        print(f"      回滚到 best 检查点（第 {best_epoch} 轮，{spec['metric']} {best_metric:.4f}）", flush=True)

    print("[5/6] 决策", flush=True)
    after = evaluate(model)
    after_pred = predict_of(model, va)
    new_v, old_v = after[spec["metric"]], before[spec["metric"]]
    gain = (new_v - old_v) if spec["higher_better"] else (old_v - new_v)

    ci = None
    if epochs == 0:
        decision = "baseline_eval"
        gain = 0.0
    elif args.gate == "point":
        delta = args.promote_delta if args.promote_delta is not None else 0.01
        thr = delta if spec["higher_better"] else delta * abs(old_v)
        decision = "promote" if gain > thr else "rollback"
    else:
        if args.bootstrap > 0:
            gains = bootstrap_gain(va["y"], before_pred, after_pred, spec["kind"],
                                   len(classes), spec["higher_better"],
                                   n_boot=args.bootstrap, seed=args.iter * 7919 + 13)
            ci = ci_summary(gains)
            decisive = ci["gain_p5"] > 0
            print(f"      增益 均值 {ci['gain_mean']:.4f}  90% 区间 [{ci['gain_p5']:.4f}, {ci['gain_p95']:.4f}]"
                  f"  P(增益>0)={ci['p_positive']:.2f}", flush=True)
        else:
            decisive = gain > 0
        decision = "promote" if decisive else "rollback"

    ood_info = None
    if ood is not None and epochs > 0:
        ood_after = evaluate(model, ood)
        drop = relative_drop(ood_before[spec["metric"]], ood_after[spec["metric"]], spec["higher_better"])
        ood_info = {"before": ood_before, "after": ood_after,
                    "relative_drop": round(drop, 4), "tolerance": args.ood_tolerance,
                    "passed": bool(drop <= args.ood_tolerance)}
        if decision == "promote" and not ood_info["passed"]:
            decision = "rollback_ood"
            print(f"      OOD 闸门否决：旧批次掉 {drop*100:.1f}% > {args.ood_tolerance*100:.0f}%", flush=True)
        else:
            print(f"      OOD 掉 {drop*100:.1f}%（容差 {args.ood_tolerance*100:.0f}%）→ {'通过' if ood_info['passed'] else '否决'}", flush=True)

    print("[6/6] 汇总", flush=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = pkg / "runs" / f"task{args.task}-iter{args.iter:02d}-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoint.npz"
    model.save(ckpt)
    digest = checkpoint_digest(ckpt)

    metrics = {
        "trainer": "v2",
        "task": args.task, "kind": spec["kind"], "iter": args.iter,
        "parent_checkpoint_sha256": parent, "checkpoint_sha256": digest,
        "parent_tag": str(parent_path.parent.name if parent_path else "scratch"),
        "continued_from_checkpoint": bool(parent),
        "epochs_this_round": epochs, "total_epochs": model.meta["epochs"],
        "output_dir": str(out),
        "param_count": model.meta["parameter_count"],
        "train_rows": len(tr["ids"]), "val_rows": len(va["ids"]), "new_rows": n_new,
        "drift": drift, "policy": pol,
        "early_stop": {"enabled": args.patience > 0, "patience": args.patience,
                       "best_epoch": int(best_epoch), "stopped_at": int(stopped_at)},
        "before": before, "after": after,
        "before_metric": before[spec["metric"]], "after_metric": after[spec["metric"]],
        "ci": ci, "ood": ood_info,
        "decision": decision, "gain": gain,
        "gate": args.gate, "bootstrap": args.bootstrap,
        "train_seconds": round(time.time() - t0, 3),
        "checkpoint_bytes": ckpt.stat().st_size,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "history.jsonl").open("w", encoding="utf-8") as fh:
        for rec in history:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if swan:
        swan.log({"decision_promote": 1 if decision.startswith("promote") else 0,
                  "gain": float(gain or 0.0), "gain_p5": float(ci["gain_p5"]) if ci else 0.0,
                  "train_seconds": metrics["train_seconds"],
                  "checkpoint_bytes": metrics["checkpoint_bytes"]}, step=max(model.meta["epochs"], 1))
        swan.finish()

    if spec["kind"] == "classification":
        print(f"      macro-F1 {before['macro_f1']:.4f} → {after['macro_f1']:.4f}   acc {after['acc']:.4f}")
        print(f"      各类 F1: {json.dumps(after['per_class_f1'], ensure_ascii=False)}")
    else:
        line = f"      RMSE {before['rmse']:.2f} → {after['rmse']:.2f}   MAE {before['mae']:.2f} → {after['mae']:.2f}"
        if "nasa" in after:
            line += f"   NASA {before['nasa']:.1f} → {after['nasa']:.1f}"
        print(line)
    print(f"      决策 {decision}  增益 {round(gain or 0.0, 4)}")
    print(f"      耗时 {metrics['train_seconds']} s  checkpoint {metrics['checkpoint_bytes']} B")
    print(f"      输出 {out}")
    print("RESULT " + json.dumps({k: v for k, v in metrics.items()
                                  if k not in ("before", "after", "drift", "policy", "ood", "ci")},
                                 ensure_ascii=False))

if __name__ == "__main__":
    main()
