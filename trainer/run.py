import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAINER = HERE / "trainer_v2.py"

TASK_HINTS = {
    "A": {"keys": {"features"}, "flat_len": 128, "pkg": "cpu-a", "name": "气体识别"},
    "B": {"keys": {"features"}, "flat_len": 34, "pkg": "cpu-b", "name": "轴承 RUL(分钟)"},
    "C": {"keys": {"settings", "sensors"}, "flat_len": 24, "pkg": "cpu-c", "name": "发动机 RUL(循环)"},
}

def read_first_record(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"空文件: {path}")

def count_flat(value, limit=512):
    n = 0
    stack = [value]
    while stack and n < limit:
        cur = stack.pop()
        if isinstance(cur, dict):
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
        elif isinstance(cur, bool):
            continue
        elif isinstance(cur, (int, float)):
            n += 1
    return n

def detect_task(path):
    path = Path(path)
    try:
        rec = read_first_record(path)
    except Exception as e:
        return None, f"读不了首行: {e}"

    inputs = rec.get("inputs")
    if not isinstance(inputs, dict):
        return None, "没有 inputs 字典"

    keys = set(inputs.keys())
    n = count_flat(inputs)

    if TASK_HINTS["C"]["keys"] <= keys:
        return "C", f"键名 {sorted(keys)} + {n} 维"
    if "features" in keys:
        if abs(n - 128) <= 2:
            return "A", f"features 扁平 {n} 维"
        if abs(n - 34) <= 2:
            return "B", f"features 嵌套 {n} 维"
        return None, f"features 维度 {n} 不匹配已知任务(128/34)"
    for t, hint in TASK_HINTS.items():
        if abs(n - hint["flat_len"]) <= 2:
            return t, f"仅维度命中 {n} 维"
    return None, f"未知结构 keys={sorted(keys)} 维数={n}"

def classify_file(path):
    name = Path(path).name.lower()
    if "valid" in name or "holdout" in name or "test" in name:
        return "val"
    if "batch" in name:
        return "extra"
    return "train"

def discover(root):
    root = Path(root)
    found = {}
    for path in sorted(root.rglob("*.jsonl")):
        if "runs" in path.parts or "swanlog" in path.parts:
            continue
        task, why = detect_task(path)
        if task is None:
            print(f"  [跳过] {path.name}: {why}")
            continue
        role = classify_file(path)
        slot = found.setdefault(task, {"train": [], "val": [], "extra": [],
                                       "evidence": f"{path.name} → {why}"})
        slot[role].append(str(path))
    return found

def plan_for(task, files, policy):
    if not files["train"]:
        return None
    train = sorted(files["train"], key=lambda p: ("public" not in p, len(p)))[0]
    val = None
    for cand in files["val"]:
        if "holdout" in cand.lower():
            val = cand
            break
    if val is None and files["val"]:
        val = files["val"][0]

    cmd = [sys.executable, str(TRAINER), "--task", task, "--policy", policy]
    extra = [p for p in files["extra"] if p != train]
    for e in extra:
        cmd += ["--extra-data", e]
    if val:
        cmd += ["--val-data", val]
    return cmd

def run_task(task, files, policy, common_args, timeout):
    cmd = plan_for(task, files, policy)
    if cmd is None:
        return {"task": task, "status": "skipped", "reason": "没找到训练数据"}
    cmd += common_args
    t0 = time.time()
    print(f"\n=== 任务 {task} ({TASK_HINTS[task]['name']}) ===")
    print("  " + " ".join(f'"{c}"' if " " in c else c for c in cmd[1:]))
    try:
        proc = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"task": task, "status": "timeout", "seconds": timeout}
    out = proc.stdout

    result = None
    for line in out.splitlines():
        if line.startswith("RESULT "):
            try:
                result = json.loads(line[7:])
            except json.JSONDecodeError:
                pass
    if proc.returncode != 0 or result is None:
        tail = "\n".join(out.strip().splitlines()[-8:])
        return {"task": task, "status": "failed", "returncode": proc.returncode,
                "seconds": round(time.time() - t0, 2), "tail": tail}

    result["status"] = "ok"
    result["seconds"] = round(time.time() - t0, 2)
    out_dir = result.get("output_dir")
    if out_dir:
        mpath = Path(out_dir) / "metrics.json"
        if mpath.is_file():
            try:
                full = json.loads(mpath.read_text(encoding="utf-8"))
                full["status"] = "ok"
                full["seconds"] = result["seconds"]
                result = full
            except Exception:
                pass
    print(f"  → {result['decision']}  gain={result['gain']:+.4f}  {result['seconds']}s")
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="扫描哪个目录找 .jsonl 数据")
    ap.add_argument("--tasks", default="", help="只跑指定任务，如 A 或 A,B")
    ap.add_argument("--policy", default="auto", choices=["auto", "off"])
    ap.add_argument("--epochs", type=int, default=None, help="显式覆盖策略轮数（0=只评估）")
    ap.add_argument("--bootstrap", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--mode", default="disabled", choices=["local", "online", "offline", "disabled"])
    ap.add_argument("--project", default="rsi-harness")
    ap.add_argument("--workspace", default=None)
    ap.add_argument("--out", default=str(HERE / "harness_run.json"))
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"扫描 {args.data_root} ...")
    found = discover(args.data_root)
    if not found:
        print("没扫到任何可识别的数据文件。")
        return 2

    wanted = [t.strip().upper() for t in args.tasks.split(",") if t.strip()] or sorted(found)
    common = []
    if args.epochs is not None:
        common += ["--epochs", str(args.epochs)]
    if args.bootstrap is not None:
        common += ["--bootstrap", str(args.bootstrap)]
    if args.patience is not None:
        common += ["--patience", str(args.patience)]
    common += ["--mode", args.mode, "--project", args.project]
    if args.workspace:
        common += ["--workspace", args.workspace]
    if args.mode == "disabled":
        common += ["--no-swan"]

    print("\n识别结果：")
    for t in sorted(found):
        f = found[t]
        print(f"  任务 {t}: train={len(f['train'])} val={len(f['val'])} extra={len(f['extra'])}"
              f"  （{f['evidence']}）")

    plan = {"root": args.data_root, "policy": args.policy,
            "detected": {t: {k: v for k, v in found[t].items() if k != "evidence"}
                         for t in found},
            "tasks": []}

    if args.dry_run:
        print("\n[dry-run] 将要执行的命令：")
        for t in wanted:
            if t not in found:
                print(f"  任务 {t}: 数据缺失，跳过")
                continue
            cmd = plan_for(t, found[t], args.policy)
            if cmd is None:
                print(f"  任务 {t}: 无训练数据，跳过")
                continue
            print("  " + " ".join(f'"{c}"' if " " in c else c for c in cmd[1:]))
        return 0

    for t in wanted:
        if t not in found:
            plan["tasks"].append({"task": t, "status": "skipped", "reason": "数据缺失"})
            continue
        plan["tasks"].append(run_task(t, found[t], args.policy, common, args.timeout))

    plan["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    Path(args.out).write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 62)
    ok = [x for x in plan["tasks"] if x.get("status") == "ok"]
    print(f"完成 {len(ok)}/{len(plan['tasks'])} 个任务，汇总写入 {args.out}")
    for x in plan["tasks"]:
        if x.get("status") == "ok":
            print(f"  {x['task']}: {x['decision']:<14} gain={x['gain']:+.4f}  "
                  f"{x['before_metric']:.4f} → {x['after_metric']:.4f}")
        else:
            print(f"  {x.get('task')}: {x.get('status')} - {x.get('reason', x.get('tail', ''))[:80]}")
    return 0 if len(ok) == len(plan["tasks"]) else 1

if __name__ == "__main__":
    sys.exit(main())
