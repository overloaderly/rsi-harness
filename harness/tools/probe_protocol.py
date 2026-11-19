import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parent
PY = sys.executable


def _run(args, stdin_text=None, env=None):
    merged = dict(os.environ)
    merged["PYTHONPATH"] = str(HARNESS / "vendor") + os.pathsep + str(HARNESS) + os.pathsep + merged.get("PYTHONPATH", "")
    merged["PYTHONIOENCODING"] = "utf-8"
    if env:
        merged.update(env)
    return subprocess.run([PY, str(HARNESS / "harness.py")] + args,
                          input=stdin_text, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=merged, cwd=str(HARNESS))


def check_describe():
    result = _run(["describe"])
    problems = []
    if result.returncode != 0:
        problems.append("describe 退出码 %d，stderr 尾部: %s" % (result.returncode, result.stderr.strip()[-400:]))
        return problems, None
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        problems.append("describe 应输出恰好 1 行帧，实际 %d 行" % len(lines))
        return problems, None
    try:
        frame = json.loads(lines[0])
    except json.JSONDecodeError as error:
        problems.append("describe 帧不是合法 JSON: %s" % error)
        return problems, None
    expected = {"protocol", "model_access", "sdk", "sdk_version", "tasks"}
    if set(frame) != expected:
        problems.append("describe 帧字段为 %s，期望 %s" % (sorted(frame), sorted(expected)))
    if frame.get("protocol") != "contest.harness.workspace.v2":
        problems.append("protocol 不是 contest.harness.workspace.v2：%r" % (frame.get("protocol"),))
    if frame.get("model_access") != "host_gateway_rpc":
        problems.append("model_access 应为 host_gateway_rpc：%r" % (frame.get("model_access"),))
    tasks = frame.get("tasks")
    if not isinstance(tasks, list) or not tasks or any(t not in ("classification", "regression") for t in tasks):
        problems.append("tasks 非法：%r" % (tasks,))
    return problems, frame


def check_predict(model_path, rows_path, limit=200):
    problems = []
    rows = []
    for line in Path(rows_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
        if len(rows) >= limit:
            break
    if not rows:
        return ["预测测试没有可用输入行"], None
    payload = "".join(json.dumps({"task": "classification", "inputs": row["inputs"]}, ensure_ascii=False) + "\n" for row in rows)
    result = _run(["predict"], stdin_text=payload, env={"CONTEST_ARTIFACT_PATH": str(model_path)})
    if result.returncode != 0:
        return ["predict 退出码 %d，stderr 尾部: %s" % (result.returncode, result.stderr.strip()[-600:])], None
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != len(rows):
        problems.append("预测行数 %d ≠ 输入行数 %d" % (len(lines), len(rows)))
    seen = 0
    for index, line in enumerate(lines[:5]):
        try:
            body = json.loads(line)
        except json.JSONDecodeError as error:
            problems.append("第 %d 行输出不是 JSON: %s" % (index, error))
            continue
        outputs = body.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            problems.append("第 %d 行 outputs 不是非空列表" % index)
            continue
        labels = [item.get("label") for item in outputs]
        scores = [item.get("score") for item in outputs]
        if len(set(labels)) != len(labels):
            problems.append("第 %d 行 label 有重复" % index)
        if any(not isinstance(s, (int, float)) or not (0 <= s <= 1) for s in scores):
            problems.append("第 %d 行 score 越界" % index)
        if scores.count(max(scores)) != 1:
            problems.append("第 %d 行最高分并列" % index)
        seen += 1
    return problems, {"inputs": len(rows), "outputs": len(lines), "sampled": seen}


def main():
    model = HARNESS.parent / "labs" / "cpu-a" / "checkpoint.npz"
    data = HARNESS.parent / "labs" / "cpu-a" / "public" / "validation.jsonl"
    if not model.is_file():
        model = HERE.parent.parent / "labs" / "cpu-a" / "checkpoint.npz"
    failures = 0
    problems, frame = check_describe()
    print("[1] describe:", "PASS" if not problems else "FAIL")
    print("    frame =", json.dumps(frame, ensure_ascii=False) if frame else "-")
    for item in problems:
        print("    -", item)
    failures += bool(problems)

    problems, stats = check_predict(model, data)
    print("[2] predict:", "PASS" if not problems else "FAIL", stats or "")
    for item in problems:
        print("    -", item)
    failures += bool(problems)

    print("RESULT", "PASS" if failures == 0 else "FAIL(%d)" % failures)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
