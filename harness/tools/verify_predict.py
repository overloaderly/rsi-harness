import json
import os
import subprocess
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HARNESS / "vendor"))

from contest_harness.predict import normalize_output  # noqa: E402


def run(checkpoint, data_path, task, limit=0):
    rows = []
    for line in Path(data_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
        if limit and len(rows) >= limit:
            break
    payload = "".join(json.dumps({"task": task, "inputs": row["inputs"]}, ensure_ascii=False) + "\n" for row in rows)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(HARNESS / "vendor") + os.pathsep + str(HARNESS)
    env["CONTEST_ARTIFACT_PATH"] = str(checkpoint)
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run([sys.executable, str(HARNESS / "harness.py"), "predict"],
                            input=payload, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", env=env, cwd=str(HARNESS))
    if result.returncode != 0:
        print("  predict 退出码 %d；stderr 尾部：%s" % (result.returncode, result.stderr.strip()[-500:]))
        return 1
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    failures = 0
    first = None
    for index, line in enumerate(lines):
        try:
            body = json.loads(line)
            normalize_output(task, body)
        except Exception as error:
            failures += 1
            if first is None:
                first = (index, "%s: %s" % (type(error).__name__, error), line[:200])
    print("  输入 %d 行 → 输出 %d 行，SDK 校验失败 %d 行" % (len(rows), len(lines), failures))
    if len(lines) != len(rows):
        failures += 1
        print("  行数不一致：缺 %d 行" % (len(rows) - len(lines)))
    if first:
        print("  首个失败：行 %d，%s" % (first[0], first[1]))
        print("  原始输出：%s" % first[2])
    return failures


def main():
    checkpoint = Path(sys.argv[1])
    data_path = Path(sys.argv[2])
    task = sys.argv[3]
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    failures = run(checkpoint, data_path, task, limit)
    print("RESULT", "PASS" if failures == 0 else "FAIL(%d)" % failures)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
