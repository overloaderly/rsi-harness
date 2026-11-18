import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIRS = {"A": "cpu-a", "B": "cpu-b", "C": "cpu-c"}

def key_of(task, rec):
    if task == "A":
        return rec["answer"]
    rid = rec["id"]
    if task == "B":
        return rid.rsplit("_s", 1)[0]
    return rid.split("_")[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["A", "B", "C"])
    ap.add_argument("--ratio", type=float, default=0.6, help="新批次里用于训练的比例")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    pkg = (Path(args.data_dir) if args.data_dir else HERE / DIRS[args.task]).resolve()
    src = pkg / "public" / "validation.jsonl"
    out_dir = pkg / "batches"
    out_dir.mkdir(exist_ok=True)

    groups = defaultdict(list)
    with src.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            groups[key_of(args.task, rec)].append(rec)

    rng = random.Random(args.seed)
    train_recs, hold_recs = [], []
    if args.task == "A":
        for k in sorted(groups, key=str):
            items = groups[k][:]
            rng.shuffle(items)
            n = max(1, int(round(len(items) * args.ratio)))
            n = min(n, len(items) - 1) if len(items) > 1 else 1
            train_recs += items[:n]
            hold_recs += items[n:]
    else:
        keys = sorted(groups, key=str)
        rng.shuffle(keys)
        n = max(1, min(int(round(len(keys) * args.ratio)), len(keys) - 1))
        train_keys = set(keys[:n])
        for k in keys:
            (train_recs if k in train_keys else hold_recs).extend(groups[k])
        print(f"  整组切分：训练组 {sorted(train_keys, key=str)}")
        print(f"             留出组 {[k for k in keys if k not in train_keys]}")

    rng.shuffle(train_recs)
    rng.shuffle(hold_recs)

    tp, hp = out_dir / "batch2_train.jsonl", out_dir / "batch2_holdout.jsonl"
    for path, recs in ((tp, train_recs), (hp, hold_recs)):
        with path.open("w", encoding="utf-8") as fh:
            for rec in recs:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"task={args.task}  源 {len(sum(groups.values(), []))} 条  ratio={args.ratio}")
    print(f"  {tp.name:24s} {len(train_recs):6d} 条  → 可训练")
    print(f"  {hp.name:24s} {len(hold_recs):6d} 条  → 只评测")
    print(f"  分层键: {len(groups)} 组")
    for k in sorted(groups, key=str):
        s = sum(1 for r in train_recs if key_of(args.task, r) == k)
        h = sum(1 for r in hold_recs if key_of(args.task, r) == k)
        print(f"    {str(k)[:32]:34s} train {s:5d} / hold {h:5d}")

if __name__ == "__main__":
    main()
