import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(os.environ.get("SELFTEST_ROOT") or (Path(tempfile.gettempdir()) / "harness-selftest"))
_DEFAULT_APP = Path("/app") if (Path("/app") / "harness.py").is_file() else Path(__file__).resolve().parent
APP = Path(os.environ.get("SELFTEST_APP") or _DEFAULT_APP)
ENTRY = APP / "harness.py"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))


def _platform_dir(env_name, platform_path, fallback):
    """平台挂载点（/previous、/model）可写时优先用它，否则退回工作目录。

    SDK 里 ParentView.root 和 predict 的 model_dir() 都写死了 /previous 与 /model，
    所以自测必须把父子产物放到同样的位置，否则会报 PATH_UNSAFE。
    """
    value = os.environ.get(env_name)
    if value:
        return Path(value)
    candidate = Path(platform_path)
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / (".selftest-probe-%d" % os.getpid())
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return candidate
    except OSError:
        return Path(fallback)
PREVIOUS = Path(os.environ.get("SELFTEST_PREVIOUS") or "/previous")
MODEL = Path(os.environ.get("SELFTEST_MODEL") or "/model")
PROTOCOL = "contest.harness.workspace.v2"
PREDICT_ARGV = ["python3", str(ENTRY), "predict"]
CLASSES_A = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]
TASK = {"A": "classification", "B": "regression", "C": "regression"}
TASK_NAME = {"A": "气体识别 · 6 类 · Macro-F1", "B": "轴承 RUL · RMSE/MAE · 分钟",
             "C": "发动机 RUL · RMSE/NASA 分 · 循环"}

DEMO = False
PRINT_LOCK = threading.Lock()

def say(text=""):
    with PRINT_LOCK:
        print(text, flush=True)

def log(message):
    say(("   [平台] " if DEMO else "[selftest] ") + str(message))

def log_round(round_index, record):
    if DEMO:
        configuration = record.get("configuration") or {}
        log("第 %d 轮完成：%s  决策 %s  候选 %s" % (
            round_index, record.get("selected_artifact_id"),
            configuration.get("decision"), fmt_score(score_of(configuration.get("validation")))))
    else:
        log("第 %d 轮完成: %s" % (round_index, json.dumps(record, ensure_ascii=False)[:400]))

def banner(title, char="─", width=68):
    say("\n" + char * width)
    say(" " + title)
    say(char * width)

def score_of(block):
    return block.get("score") if isinstance(block, dict) else None

def fmt_score(value):
    return "n/a" if not isinstance(value, (int, float)) else "%.4f" % value

def sha(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()

def synthetic(task_key, count, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(count):
        if task_key == "A":
            label = CLASSES_A[index % len(CLASSES_A)]
            base = rng.normal(0, 1, 128)
            base[CLASSES_A.index(label)] += 2.5
            inputs = {"features": [round(float(v), 6) for v in base]}
            answer = label
        elif task_key == "B":
            horizontal = [round(float(v), 6) for v in rng.normal(0, 1, 17)]
            vertical = [round(float(v), 6) for v in rng.normal(0, 1, 17)]
            inputs = {"features": {"horizontal": horizontal, "vertical": vertical}}
            answer = float(max(0.0, 120.0 - index))
        else:
            inputs = {"settings": [round(float(v), 6) for v in rng.normal(0, 1, 3)],
                      "sensors": [round(float(v), 6) for v in rng.normal(0, 1, 21)]}
            answer = float(max(0.0, 200.0 - index))
        rows.append({"id": "synthetic-%04d" % index, "inputs": inputs, "answer": answer})
    return rows

def load_rows(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows

def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

def build_arrays(rows, task_key):
    sys.path.insert(0, str(APP))
    from rsi_data import flatten
    ids, features, labels, paths = [], [], [], None
    for row in rows:
        current, values = flatten(row["inputs"])
        if paths is None:
            paths = current
        elif current != paths:
            raise RuntimeError("synthetic schema changed")
        ids.append(row["id"])
        features.append(values)
        labels.append(CLASSES_A.index(row["answer"]) if task_key == "A" else float(row["answer"]))
    import numpy as np
    dtype = np.int64 if task_key == "A" else np.float64
    return ids, np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=dtype), paths

class Host:
    def __init__(self, demo=False):
        self.candidate = None
        self.state = None
        self.records = []
        self.calls = 0
        self.demo = demo

    def answer(self, frame):
        if not self.demo:
            return os.environ.get("SELFTEST_DECISION", "train")
        payload = {}
        try:
            text = "".join(str(item.get("content") or "") for item in frame.get("messages") or [])
            begin, end = text.find("{"), text.rfind("}")
            if begin >= 0 and end > begin:
                payload = json.loads(text[begin:end + 1])
        except (ValueError, TypeError):
            payload = {}
        history = payload.get("history") or []
        task = payload.get("task")
        parent = score_of(payload.get("parent_validation"))
        planned = payload.get("planned_epochs")
        if not history:
            say("   ◂ 决策桩：history 为空（首轮），父基线=%s → train（按基准配置完整训练）"
                  % fmt_score(parent))
            return "train"
        if len(history) < 2:
            say("   ◂ 决策桩：历史只有 %d 条，还看不出趋势 → train（沿用基准配置继续训练）"
                  % len(history))
            return "train"
        last = score_of(history[-1].get("validation"))
        previous = score_of(history[-2].get("validation"))
        if not isinstance(last, (int, float)) or not isinstance(previous, (int, float)):
            say("   ◂ 决策桩：历史缺少可比较指标 → train（保守沿用基准配置）")
            return "train"
        improved = last > previous if task == "classification" else last < previous
        if improved:
            say("   ◂ 决策桩：上轮=%s 好于前一轮=%s （规划 %s epoch）→ conserve（小步微调，避免过拟合）"
                  % (fmt_score(last), fmt_score(previous), planned))
            return "conserve"
        say("   ◂ 决策桩：上轮=%s 没有好于前一轮=%s → train（换回常规步长重新探索）"
            % (fmt_score(last), fmt_score(previous)))
        return "train"

    def _pump(self, stream):
        for line in stream:
            text = line.rstrip()
            if text:
                say("   ▸ " + text)
        stream.close()

    def serve_train(self, start, history, timeout=900):
        process = subprocess.Popen(
            [sys.executable, str(ENTRY), "train"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1)
        threading.Thread(target=self._pump, args=(process.stderr,), daemon=True).start()
        process.stdin.write(json.dumps(start, ensure_ascii=False) + "\n")
        process.stdin.flush()
        sequence, started = 0, time.monotonic()
        record = None
        while True:
            if time.monotonic() - started > timeout:
                process.kill()
                raise RuntimeError("train 超时")
            line = process.stdout.readline()
            if not line:
                raise RuntimeError("participant closed stdout (exit=%s)" % process.poll())
            frame = json.loads(line)
            kind = frame.get("type")
            if kind == "round_failed":
                process.kill()
                raise RuntimeError("round_failed: %s" % frame.get("error"))
            if kind == "round_complete":
                record = frame["record"]
                self.records.append(record)
                process.stdin.write(json.dumps({"type": "round_ack", "accepted": True}) + "\n")
                process.stdin.flush()
                process.stdin.close()
                break
            if frame.get("id") != sequence:
                raise RuntimeError("rpc 顺序错误: %r" % frame)
            sequence += 1
            if kind == "history_request":
                reply = {"type": "history_response", "id": frame["id"], "records": history}
            elif kind == "model_request":
                self.calls += 1
                content = self.answer(frame)
                reply = {"type": "model_response", "id": frame["id"], "call_id": "call-%d" % self.calls,
                         "response": {"choices": [{"message": {"role": "assistant", "content": content}}]}}
            elif kind == "state_request":
                self.state = {item["path"]: base64.b64decode(item["data_base64"]) for item in frame.get("files") or []}
                reply = {"type": "state_response", "id": frame["id"], "state_sha256": sha(b"state")}
            elif kind == "candidate_request":
                self.candidate = {"artifact": frame["artifact"],
                                  "files": {item["path"]: base64.b64decode(item["data_base64"])
                                            for item in frame.get("files") or []}}
                reply = {"type": "candidate_response", "id": frame["id"],
                         "artifact_id": "cand-%d" % len(self.records), "summary": {"metrics": {}}}
            else:
                raise RuntimeError("未知帧 %r" % kind)
            process.stdin.write(json.dumps(reply, ensure_ascii=False) + "\n")
            process.stdin.flush()
        code = process.wait(timeout=60)
        if code != 0:
            raise RuntimeError("train 退出码 %s" % code)
        return record

    def materialize_parent(self):
        target = PREVIOUS
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        for name, raw in self.candidate["files"].items():
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        return [{"path": name, "sha256": sha(raw), "size_bytes": len(raw)}
                for name, raw in sorted(self.candidate["files"].items())]

def start_frame(task_key, round_index, parent_files, parent_digest, artifacts, data_dir, state_dir):
    return {
        "protocol": PROTOCOL,
        "prompt": "自测用固定提示：按公开数据续训参考模型，并在每轮结束提交候选产物。",
        "prompt_version": "industrial-workspace-flexible-v3",
        "dataset_key": task_key,
        "round_index": round_index,
        "stage_id": "selftest",
        "task_contract": {"kind": TASK[task_key], "classes": CLASSES_A if task_key == "A" else [],
                          "public_projection": {"train_path": str(data_dir / "train.jsonl"),
                                                "validation_path": str(data_dir / "validation.jsonl")}},
        "model_name": "self-hosted-llm",
        "max_output_tokens": 64,
        "parent_checkpoint_sha256": parent_digest,
        "parent_artifact": {"artifact_id": "parent-%d" % (round_index - 1),
                            "artifact": {"checkpoint_paths": ["checkpoint.npz"], "code_paths": [],
                                         "predict_argv": PREDICT_ARGV},
                            "files": parent_files},
        "artifacts": artifacts,
        "state_path": str(state_dir) if state_dir else None,
        "state_sha256": sha(b"state"),
        "wall_seconds": 600,
        "resource_limits": {"cpu": 2, "memory_mb": 2048},
    }

def predict_check(task_key, checkpoint, sample_inputs):
    model_dir = MODEL
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True)
    shutil.copyfile(checkpoint, model_dir / "checkpoint.npz")
    context = {"protocol": "contest.predict-context.v1", "dataset_key": task_key,
               "task": TASK[task_key], "artifact": {}, "artifact_id": "cand-x"}
    env = dict(os.environ)
    env["CONTEST_PREDICTION_CONTEXT"] = json.dumps(context, ensure_ascii=False)
    lines = [json.dumps({"task": TASK[task_key], "inputs": item}, ensure_ascii=False) for item in sample_inputs]
    finished = subprocess.run([sys.executable, str(ENTRY), "predict"],
                              input="\n".join(lines) + "\n", capture_output=True, text=True, env=env)
    if finished.returncode != 0:
        raise RuntimeError("predict 失败: %s" % finished.stderr[-2000:])
    outputs = [json.loads(line) for line in finished.stdout.splitlines() if line.strip()]
    if len(outputs) != len(lines):
        raise RuntimeError("predict 输出行数不符: %d != %d" % (len(outputs), len(lines)))
    sys.path.insert(0, str(APP))
    from contest_harness.predict import normalize_output
    for item in outputs:
        normalize_output(TASK[task_key], item)
    return outputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="A", choices=["A", "B", "C"])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--parent", default=None, help="直接用给定的父检查点（例如 labs 里的官方基线）")
    parser.add_argument("--train-rows", type=int, default=200)
    parser.add_argument("--val-rows", type=int, default=0, help="验证集行数上限，0 = 用整份")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--demo", action="store_true", help="分轮横幅 + 讲解式输出（录演示视频用）")
    parser.add_argument("--pause", action="store_true", help="每轮结束后等待回车（录屏节奏用）")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    global DEMO
    DEMO = bool(args.demo)
    started = time.monotonic()
    if args.demo:
        banner("智训引擎 Harness 离线演示 · 任务 %s（%s）" % (args.task, TASK_NAME[args.task]), "═")

    if ROOT.exists() and not args.keep:
        shutil.rmtree(ROOT)
    global PREVIOUS, MODEL
    PREVIOUS = _platform_dir("SELFTEST_PREVIOUS", "/previous", ROOT / "previous")
    MODEL = _platform_dir("SELFTEST_MODEL", "/model", ROOT / "model")
    PREVIOUS.mkdir(parents=True, exist_ok=True)
    MODEL.mkdir(parents=True, exist_ok=True)
    data_dir = ROOT / "data"
    if args.data_dir:
        source = Path(args.data_dir)
        train = load_rows(source / "train.jsonl")[: args.train_rows]
        validate = load_rows(source / "validation.jsonl")
        if args.val_rows:
            validate = validate[:args.val_rows]
    else:
        train = synthetic(args.task, args.train_rows, 7)
        validate = synthetic(args.task, args.val_rows or max(30, args.train_rows // 3), 13)
    write_rows(data_dir / "train.jsonl", train)
    write_rows(data_dir / "validation.jsonl", validate)
    log("数据: train=%d validation=%d (task %s)" % (len(train), len(validate), args.task))

    ids, x, y, paths = build_arrays(train, args.task)
    log("特征 %d 项，参数预算 %d" % (len(paths), len(paths) * 32 + 32 + (len(CLASSES_A) if args.task == "A" else 1) * 32))

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from rsi_model import Model
    parent_dir = ROOT / "parent"
    parent_dir.mkdir(parents=True, exist_ok=True)
    if args.parent:
        shutil.copyfile(args.parent, parent_dir / "checkpoint.npz")
        log("父检查点来自 --parent: %s" % args.parent)
    else:
        model = Model(TASK[args.task], paths, CLASSES_A if args.task == "A" else [], x, y,
                      hidden=min(32, 96), seed=42, learning_rate=0.003, batch_size=64)
        model.save(parent_dir / "checkpoint.npz")
    parent_raw = (parent_dir / "checkpoint.npz").read_bytes()
    parent_files = [{"path": "checkpoint.npz", "sha256": sha(parent_raw), "size_bytes": len(parent_raw)}]
    parent_digest = sha(parent_raw)
    log("父检查点 %d 字节" % len(parent_raw))

    previous = PREVIOUS
    if previous.exists():
        shutil.rmtree(previous)
    previous.mkdir(parents=True)
    (previous / "checkpoint.npz").write_bytes(parent_raw)

    host = Host(demo=args.demo)
    history = []
    state_dir = None
    artifacts = []
    artifact_id = None
    summary = []
    for round_index in range(1, args.rounds + 1):
        if round_index > 1:
            parent_files = host.materialize_parent()
            parent_digest = sha((PREVIOUS / parent_files[0]["path"]).read_bytes())
            artifacts = [{"artifact_id": artifact_id, "artifact": host.candidate["artifact"],
                          "files": parent_files}]
            state_dir = ROOT / "state-1"
        if args.demo:
            banner("第 %d 轮 / 共 %d 轮 · 父产物 %s" % (round_index, args.rounds, artifact_id or "官方基线"))
        record = host.serve_train(start_frame(args.task, round_index, parent_files, parent_digest,
                                              artifacts, data_dir, state_dir), history)
        configuration = record.get("configuration") or {}
        summary.append({"round": round_index,
                        "parent": configuration.get("parent_validation"),
                        "current": configuration.get("validation"),
                        "decision": configuration.get("decision"),
                        "artifact": record.get("selected_artifact_id"),
                        "reason": record.get("reason")})
        log_round(round_index, record)
        artifact_id = record.get("selected_artifact_id") or artifact_id
        history = history + [{"id": "round-%d" % round_index,
                              "hypothesis": record.get("hypothesis"),
                              "configuration": record.get("configuration"),
                              "reason": record.get("reason")}]
        if args.demo and round_index < args.rounds:
            if args.pause:
                try:
                    input("\n  ⏸  已暂停，按回车进入第 %d 轮…… " % (round_index + 1))
                except EOFError:
                    pass
        if round_index == 1 and host.state:
            state_dir = ROOT / "state-1"
            state_dir.mkdir(parents=True, exist_ok=True)
            for name, raw in host.state.items():
                (state_dir / name).write_bytes(raw)
            log("state 回传: %s" % sorted(host.state))

    if not host.candidate:
        raise RuntimeError("没有收到候选产物")
    checkpoint = ROOT / "candidate" / "checkpoint.npz"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(host.candidate["files"]["checkpoint.npz"])

    samples = [row["inputs"] for row in validate[:3]]
    outputs = predict_check(args.task, checkpoint, samples)
    if DEMO:
        head = outputs[0]["outputs"] if outputs else None
        if isinstance(head, list) and head:
            best = max(head, key=lambda item: item["score"])
            log("predict 输出: %d 条样本，首条 top-1 = %s（score %.3f），共 %d 个类别候选"
                % (len(outputs), best["label"], best["score"], len(head)))
        elif isinstance(head, dict):
            log("predict 输出: %d 条样本，首条 value = %.2f（非负有限数）"
                % (len(outputs), head["value"]))
    else:
        log("predict 输出: %s" % json.dumps(outputs, ensure_ascii=False)[:400])

    if args.demo:
        banner("实验台账 · %d 轮汇总" % args.rounds, "═")
        for item in summary:
            delta = ""
            parent, current = score_of(item["parent"]), score_of(item["current"])
            if isinstance(parent, (int, float)) and isinstance(current, (int, float)):
                diff = current - parent
                better = diff > 0 if TASK[args.task] == "classification" else diff < 0
                delta = "  (%+.4f%s)" % (diff, " 采纳" if better else " 回滚/持平")
            say(" 第 %d 轮  决策=%-8s 父=%s → 候选=%s%s" % (
                item["round"], item["decision"], fmt_score(parent), fmt_score(current), delta))
        say("\n 用时 %.1fs · 全程无人工干预 · 每轮 1 次模型调用" % (time.monotonic() - started))
        say(" ledger.json 记录 %d 轮（读历史 → 做实验 → 依据结果决策 → 提交/回滚）" % args.rounds)

    log("离线自测通过")

if __name__ == "__main__":
    main()
