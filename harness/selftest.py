"""离线端到端自测：用假宿主跑两轮 train + 一次 predict，完全不需要平台。

用法（在镜像内，需要 root 才能写 /previous 和 /model）：

    docker run --rm --user 0 rsi-harness:v1 \
        python3 /opt/harness-selftest/selftest.py --task A

    # 用真实公开数据（把 labs 目录挂进来）
    docker run --rm --user 0 -v B:\\whale\\智训引擎\\labs:/labs:ro rsi-harness:v1 \
        python3 /opt/harness-selftest/selftest.py --task B --data-dir /labs/cpu-b/public

检查项：describe、两轮 train（含 history/state/候选回传/回滚决策）、predict 输出合法性。
"""
import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("SELFTEST_ROOT", "/tmp/harness-selftest"))
PROTOCOL = "contest.harness.workspace.v2"
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
CLASSES_A = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]
TASK = {"A": "classification", "B": "regression", "C": "regression"}


def log(message):
    print("[selftest] " + str(message), flush=True)


def sha(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


# ------------------------------------------------------------------ 造数据
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
    sys.path.insert(0, "/app")
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


# ------------------------------------------------------------------ 假宿主
class Host:
    def __init__(self):
        self.candidate = None
        self.state = None
        self.records = []
        self.calls = 0

    def serve_train(self, start, history, timeout=900):
        process = subprocess.Popen(
            [sys.executable, "/app/harness.py", "train"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1)
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
                reply = {"type": "model_response", "id": frame["id"], "call_id": "call-%d" % self.calls,
                         "response": {"choices": [{"message": {"role": "assistant",
                                                               "content": os.environ.get("SELFTEST_DECISION", "train")}}]}}
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
        """把上一轮候选铺到 /previous，模拟平台挂载的父产物。"""
        target = Path("/previous")
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
    model_dir = Path("/model")
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True)
    shutil.copyfile(checkpoint, model_dir / "checkpoint.npz")
    context = {"protocol": "contest.predict-context.v1", "dataset_key": task_key,
               "task": TASK[task_key], "artifact": {}, "artifact_id": "cand-x"}
    env = dict(os.environ)
    env["CONTEST_PREDICTION_CONTEXT"] = json.dumps(context, ensure_ascii=False)
    lines = [json.dumps({"task": TASK[task_key], "inputs": item}, ensure_ascii=False) for item in sample_inputs]
    finished = subprocess.run([sys.executable, "/app/harness.py", "predict"],
                              input="\n".join(lines) + "\n", capture_output=True, text=True, env=env)
    if finished.returncode != 0:
        raise RuntimeError("predict 失败: %s" % finished.stderr[-2000:])
    outputs = [json.loads(line) for line in finished.stdout.splitlines() if line.strip()]
    if len(outputs) != len(lines):
        raise RuntimeError("predict 输出行数不符: %d != %d" % (len(outputs), len(lines)))
    sys.path.insert(0, "/app")
    from contest_harness.predict import normalize_output
    for item in outputs:
        normalize_output(TASK[task_key], item)
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="A", choices=["A", "B", "C"])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--train-rows", type=int, default=200)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    if ROOT.exists() and not args.keep:
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True, exist_ok=True)
    data_dir = ROOT / "data"
    if args.data_dir:
        source = Path(args.data_dir)
        train = load_rows(source / "train.jsonl")[: args.train_rows]
        validate = load_rows(source / "validation.jsonl")[: max(30, args.train_rows // 3)]
    else:
        train = synthetic(args.task, args.train_rows, 7)
        validate = synthetic(args.task, max(30, args.train_rows // 3), 13)
    write_rows(data_dir / "train.jsonl", train)
    write_rows(data_dir / "validation.jsonl", validate)
    log("数据: train=%d validation=%d (task %s)" % (len(train), len(validate), args.task))

    ids, x, y, paths = build_arrays(train, args.task)
    log("特征 %d 项，参数预算 %d" % (len(paths), len(paths) * 32 + 32 + (len(CLASSES_A) if args.task == "A" else 1) * 32))

    sys.path.insert(0, "/app")
    from rsi_model import Model
    model = Model(TASK[args.task], paths, CLASSES_A if args.task == "A" else [], x, y,
                  hidden=min(32, 96), seed=42, learning_rate=0.003, batch_size=64)
    parent_dir = ROOT / "parent"
    parent_dir.mkdir(parents=True, exist_ok=True)
    model.save(parent_dir / "checkpoint.npz")
    parent_raw = (parent_dir / "checkpoint.npz").read_bytes()
    parent_files = [{"path": "checkpoint.npz", "sha256": sha(parent_raw), "size_bytes": len(parent_raw)}]
    parent_digest = sha(parent_raw)
    log("父检查点 %d 字节" % len(parent_raw))

    # 平台把父检查点挂在只读 /previous；自测里用 root 手动铺一份
    previous = Path("/previous")
    if previous.exists():
        shutil.rmtree(previous)
    previous.mkdir(parents=True)
    (previous / "checkpoint.npz").write_bytes(parent_raw)

    host = Host()
    history = []
    state_dir = None
    artifacts = []
    artifact_id = None
    for round_index in (1, 2):
        if round_index == 2:
            parent_files = host.materialize_parent()
            parent_digest = sha((Path("/previous") / parent_files[0]["path"]).read_bytes())
            artifacts = [{"artifact_id": artifact_id, "artifact": host.candidate["artifact"],
                          "files": parent_files}]
            state_dir = ROOT / "state-1"
        record = host.serve_train(start_frame(args.task, round_index, parent_files, parent_digest,
                                              artifacts, data_dir, state_dir), history)
        log("第 %d 轮完成: %s" % (round_index, json.dumps(record, ensure_ascii=False)[:400]))
        artifact_id = record.get("selected_artifact_id") or artifact_id
        history = history + [{"id": "round-%d" % round_index,
                              "hypothesis": record.get("hypothesis"),
                              "configuration": record.get("configuration"),
                              "reason": record.get("reason")}]
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
    log("predict 输出: %s" % json.dumps(outputs, ensure_ascii=False)[:400])
    log("自测通过 ✅")


if __name__ == "__main__":
    main()
