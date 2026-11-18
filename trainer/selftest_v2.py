import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PY = sys.executable
FAILED = []

def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""), flush=True)
    if not ok:
        FAILED.append(name)

def load_trainer():
    spec = importlib.util.spec_from_file_location("trainer_v2", HERE / "trainer_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_psi(mod):
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 2000)
    v_same = mod.psi(a, rng.normal(0, 1, 2000))
    v_shift = mod.psi(a, rng.normal(3, 1, 2000))
    check("PSI 同分布应接近 0", v_same < 0.02, f"psi={v_same:.4f}")
    check("PSI 大幅偏移应远大于 0.25", v_shift > 1.0, f"psi={v_shift:.4f}")
    ref = np.array([0.0] * 50 + [1.0] * 50)
    cur = np.array([0.0] * 25 + [1.0] * 75)
    got = mod.psi(ref, cur, bins=2)
    exp = (0.25 - 0.5) * np.log(0.25 / 0.5) + (0.75 - 0.5) * np.log(0.75 / 0.5)
    check("PSI 与手算一致（两箱）", abs(got - exp) < 1e-9, f"{got:.6f} vs {exp:.6f}")

def test_bootstrap(mod):
    rng = np.random.default_rng(1)
    n = 800
    y = rng.uniform(0, 100, n)
    before = y + rng.normal(0, 10, n)
    after_same = before.copy()
    after_better = y + rng.normal(0, 5, n)

    g_same = mod.bootstrap_gain(y, before, after_same, "regression", 0, False, n_boot=300, seed=7)
    check("同一模型自比：增益恒为 0", float(np.max(np.abs(g_same))) < 1e-12,
          f"max|g|={float(np.max(np.abs(g_same))):.2e}")

    g = mod.bootstrap_gain(y, before, after_better, "regression", 0, False, n_boot=300, seed=7)
    ci = mod.ci_summary(g)
    check("明显更好：区间下界 > 0", ci["gain_p5"] > 0,
          f"p5={ci['gain_p5']:.3f} p95={ci['gain_p95']:.3f}")
    check("区间有序 p5 < p50 < p95",
          ci["gain_p5"] < ci["gain_p50"] < ci["gain_p95"])

    y_cls = rng.integers(0, 4, n)
    pb = np.eye(4)[(y_cls + 1) % 4]
    pa = np.eye(4)[y_cls]
    gc = mod.bootstrap_gain(y_cls, pb, pa, "classification", 4, True, n_boot=200, seed=3)
    check("分类：全错→全对增益为 1", abs(float(np.mean(gc)) - 1.0) < 1e-9,
          f"mean={float(np.mean(gc)):.4f}")

def test_policy(mod):
    p_hold = mod.policy_from_drift(0.05, 0.2, None, None)
    p_small = mod.policy_from_drift(0.15, 0.3, None, None)
    p_big = mod.policy_from_drift(0.60, 2.0, None, None)
    p_manual = mod.policy_from_drift(0.60, 2.0, 7, None)
    check("PSI 0.05 → 不重训", p_hold["action"] == "hold_no_retrain" and p_hold["epochs"] == 0)
    check("PSI 0.15 → 小步 3 轮 lr 0.001",
          p_small["epochs"] == 3 and p_small["lr"] == 0.001, f"{p_small['action']}")
    check("PSI 0.60 → 大步 12 轮、默认不重加权",
          p_big["epochs"] == 12 and p_big["oversample_new"] == 1, f"{p_big['action']}")
    check("显式 --epochs 覆盖策略轮数", p_manual["epochs"] == 7)

def run(extra, timeout=300):
    cmd = [PY, str(HERE / "trainer_v2.py")] + extra
    return subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)

def result_of(proc):
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return None

def test_regression(tasks_expect, quick):
    for task, (metric, expect) in tasks_expect.items():
        p = run(["--task", task, "--epochs", "0", "--iter", "1",
                 "--val-data", "batches/batch2_holdout.jsonl", "--no-swan"])
        res = result_of(p)
        ok = res is not None and abs(res["before_metric"] - expect) < 1e-2
        check(f"{task} 基线复现（{metric} ≈ {expect}）", ok,
              f"实得 {res['before_metric']:.4f}" if res else "N/A")

def test_boundaries():
    p = run(["--task", "A", "--epochs", "0", "--no-swan",
             "--from-baseline", "runs/does-not-exist.npz"])
    check("父检查点不存在 → 非零退出并报错",
          p.returncode != 0 and "父检查点不存在" in (p.stdout + p.stderr))

    p = run(["--task", "A", "--epochs", "0", "--no-swan", "--bootstrap", "0"])
    check("bootstrap=0 仍可运行", p.returncode == 0)

    p = run(["--task", "Z", "--epochs", "0", "--no-swan"])
    check("非法任务名 → 非零退出", p.returncode != 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    mod = load_trainer()
    print("=== 单元级：指标与统计 ===", flush=True)
    test_psi(mod)
    test_bootstrap(mod)
    test_policy(mod)

    print("\n=== 回归级：v2 必须复现 v1 的基线 ===", flush=True)
    if args.quick:
        print("(跳过，--quick)", flush=True)
    else:
        test_regression({"A": ("macro_f1", 0.6972),
                         "B": ("rmse", 710.81),
                         "C": ("rmse", 58.44)}, args.quick)

    print("\n=== 边界级：异常输入 ===", flush=True)
    test_boundaries()

    print("\n" + ("全部通过" if not FAILED else f"失败 {len(FAILED)} 项：{FAILED}"), flush=True)
    return 1 if FAILED else 0

if __name__ == "__main__":
    sys.exit(main())
