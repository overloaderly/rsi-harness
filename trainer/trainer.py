import argparse
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

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["A", "B", "C"])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--from-baseline", nargs="?", const="official", default="official",
                    help="父检查点：不传值=包内 checkpoint.npz；'none'=从零；或给具体路径")
    ap.add_argument("--from-last", choices=["latest", "best"], default=None,
                    help="latest=接上一轮；best=接当前 champion（官方基线也参与比较）")
    ap.add_argument("--epochs", type=int, default=None)
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
    ap.add_argument("--val-data", default=None, help="改用它做验证集（如 batches/batch2_holdout.jsonl）")
    ap.add_argument("--extra-data", action="append", default=[],
                    help="并入训练集的新批次文件，可多次指定（如 batches/batch2_train.jsonl）")
    ap.add_argument("--promote-delta", type=float, default=None)
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
    lr = args.lr if args.lr is not None else float(cfg.get("learning_rate", 0.003))
    batch = args.batch if args.batch is not None else int(cfg.get("batch_size", 128))
    epochs = args.epochs if args.epochs is not None else int(cfg.get("epochs", 15))
    classes = spec["classes"]

    t0 = time.time()
    print(f"[1/5] 读数据  task={args.task}  包={pkg.name}", flush=True)
    kwargs = {"classes": classes} if spec["kind"] == "classification" else {}
    tr = read_dataset(pkg / "public" / "train.jsonl", spec["kind"], **kwargs)
    if args.extra_data:
        xs, ys, ids = [tr["x"]], [tr["y"]], list(tr["ids"])
        for extra in args.extra_data:
            ep = Path(extra)
            if not ep.is_absolute():
                ep = (pkg / ep).resolve()
            ex = read_dataset(ep, spec["kind"], expected_paths=tr["paths"], **kwargs)
            xs.append(ex["x"]); ys.append(ex["y"]); ids += list(ex["ids"])
            print(f"      + 并入新批次 {ep.name}：{len(ex['ids'])} 条", flush=True)
        tr = {"x": np.vstack(xs), "y": np.concatenate(ys), "ids": ids, "paths": tr["paths"]}
    if args.val_data:
        vp = Path(args.val_data)
        if not vp.is_absolute():
            vp = (pkg / vp).resolve()
        va = read_dataset(vp, spec["kind"], expected_paths=tr["paths"], **kwargs)
    else:
        va = read_dataset(pkg / "public" / "validation.jsonl", spec["kind"],
                          expected_paths=tr["paths"], **kwargs)
    print(f"      train {len(tr['ids'])} / val {len(va['ids'])} / 特征 {len(tr['paths'])} 维", flush=True)

    def evaluate(m):
        pred = m.predict(va["x"])
        if spec["kind"] == "classification":
            yp = np.argmax(pred, axis=1)
            return {"acc": float(np.mean(yp == va["y"])),
                    "macro_f1": macro_f1(va["y"], yp, len(classes)),
                    "per_class_f1": per_class_f1(va["y"], yp, classes)}
        rmse, mae = rmse_mae(va["y"], pred)
        out = {"rmse": rmse, "mae": mae}
        if args.task == "C":
            out["nasa"] = nasa_score(va["y"], pred)
        return out

    print("[2/5] 漂移检测 PSI", flush=True)
    per_psi = np.array([psi(tr["x"][:, i], va["x"][:, i]) for i in range(tr["x"].shape[1])])
    drift = {"psi_mean": float(per_psi.mean()), "psi_max": float(per_psi.max()),
             "psi_frac_gt_0_25": float((per_psi > 0.25).mean())}
    print(f"      均值 {drift['psi_mean']:.3f} / 最大 {drift['psi_max']:.3f} "
          f"/ {drift['psi_frac_gt_0_25']*100:.0f}% 维度 > 0.25", flush=True)

    official = pkg / "checkpoint.npz"
    runs_dir = pkg / "runs"
    run_dirs = sorted([p for p in runs_dir.glob(f"task{args.task}-iter*-*")
                       if (p / "checkpoint.npz").is_file()], key=lambda p: p.stat().st_mtime) if runs_dir.is_dir() else []

    print("[3/5] 选父检查点 / 建模型", flush=True)
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
                      hidden=hidden, seed=seed, learning_rate=lr, batch_size=batch)
        print("      从零训练", flush=True)
    print(f"      参数量 {model.meta['parameter_count']}", flush=True)

    before = evaluate(model)
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
                                 "param_count": model.meta["parameter_count"]})
            swan = swanlab
            print(f"      SwanLab 已连接（{args.mode}）", flush=True)
        except Exception as exc:
            print(f"      SwanLab 不可用，降级为纯本地: {exc}", flush=True)

    print(f"[4/5] 训练  {epochs} 轮", flush=True)
    history = []
    base_epoch = model.meta["epochs"]
    for ep in range(1, epochs + 1):
        loss = model.fit(tr["x"], tr["y"], 1)[-1]
        s = evaluate(model)
        history.append({"epoch": base_epoch + ep, "loss": float(loss),
                        **{k: v for k, v in s.items() if k != "per_class_f1"}})
        if swan and (ep % 5 == 0 or ep == epochs):
            log = {"loss": float(loss), "psi_mean": drift["psi_mean"], "psi_max": drift["psi_max"]}
            if spec["kind"] == "classification":
                log.update({"val_acc": s["acc"], "val_macro_f1": s["macro_f1"]})
                for name, v in s["per_class_f1"].items():
                    log[f"f1_{name}"] = v
            else:
                log.update({"val_rmse": s["rmse"], "val_mae": s["mae"]})
                if "nasa" in s:
                    log["val_nasa"] = s["nasa"]
            swan.log(log, step=base_epoch + ep)
        print(f"      epoch {base_epoch + ep:3d}  loss {loss:8.4f}  {spec['metric']} {s[spec['metric']]:9.4f}", flush=True)

    print("[5/5] 汇总", flush=True)
    after = evaluate(model)
    new_v, old_v = after[spec["metric"]], before[spec["metric"]]
    if epochs == 0:
        decision, gain = "baseline_eval", 0.0
    else:
        gain = (new_v - old_v) if spec["higher_better"] else (old_v - new_v)
        delta = args.promote_delta if args.promote_delta is not None else 0.01
        thr = delta if spec["higher_better"] else delta * abs(old_v)
        decision = "promote" if gain > thr else "rollback"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = pkg / "runs" / f"task{args.task}-iter{args.iter:02d}-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoint.npz"
    model.save(ckpt)
    digest = checkpoint_digest(ckpt)

    metrics = {
        "task": args.task, "kind": spec["kind"], "iter": args.iter,
        "parent_checkpoint_sha256": parent, "checkpoint_sha256": digest,
        "parent_tag": str(parent_path.parent.name if parent_path else "scratch"),
        "continued_from_checkpoint": bool(parent),
        "epochs_this_round": epochs, "total_epochs": model.meta["epochs"],
        "param_count": model.meta["parameter_count"],
        "train_rows": len(tr["ids"]), "val_rows": len(va["ids"]),
        "drift": drift, "before": before, "after": after,
        "decision": decision, "gain": gain,
        "train_seconds": round(time.time() - t0, 3),
        "checkpoint_bytes": ckpt.stat().st_size,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "history.jsonl").open("w", encoding="utf-8") as fh:
        for rec in history:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if swan:
        swan.log({"decision_promote": 1 if decision == "promote" else 0,
                  "gain": float(gain or 0.0), "train_seconds": metrics["train_seconds"],
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
    print("RESULT " + json.dumps({k: v for k, v in metrics.items() if k not in ("before", "after", "drift")},
                                 ensure_ascii=False))

if __name__ == "__main__":
    main()
