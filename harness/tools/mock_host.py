import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROTOCOL = "contest.harness.workspace.v2"
ENTRY = ["python3", "/app/harness.py"]


def _entry():
    override = os.environ.get("MOCK_ENTRY")
    if override:
        return [part for part in override.split("|") if part]
    return list(ENTRY)


def _sha(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _files(directory):
    items = []
    for path in sorted(Path(directory).rglob("*")):
        if path.is_file():
            items.append({"path": path.relative_to(directory).as_posix(),
                          "size_bytes": path.stat().st_size,
                          "sha256": _sha(path)})
    return items


def _start_frame(dataset, train_path, validation_path, parent_dir, kind, classes, max_tokens):
    checkpoint = Path(parent_dir) / "checkpoint.npz"
    if not checkpoint.is_file():
        candidates = sorted(Path(parent_dir).rglob("*.npz"))
        if not candidates:
            raise SystemExit("父检查点缺失: %s" % parent_dir)
        checkpoint = candidates[0]
    manifest = [{"path": checkpoint.name, "size_bytes": checkpoint.stat().st_size,
                 "sha256": _sha(checkpoint)}]
    artifact = {"checkpoint_paths": [checkpoint.name], "code_paths": [],
                "predict_argv": ENTRY + ["predict"]}
    parent = {"artifact_id": "artifact-parent", "artifact": artifact, "files": manifest,
              "artifact_sha256": "sha256:" + hashlib.sha256(json.dumps(
                  {"files": manifest, "artifact": artifact}, allow_nan=False,
                  sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    contract = {"kind": kind, "classes": classes,
                "public_projection": {"train_path": train_path, "validation_path": validation_path}}
    return {
        "protocol": PROTOCOL, "prompt": "官方提示：请完成本轮续训并把过程写进记录。",
        "prompt_version": "v1", "dataset_key": dataset, "round_index": 1,
        "stage_id": "stage-1", "task_contract": contract, "model_name": "Qwen/Qwen3.5-4B",
        "max_output_tokens": max_tokens, "wall_seconds": 600,
        "parent_checkpoint_sha256": manifest[0]["sha256"], "parent_artifact": parent,
        "artifacts": [], "state_path": os.environ.get("MOCK_STATE", "/state"), "state_sha256": "",
    }


class Host:
    def __init__(self, child, candidate_root):
        self.child = child
        self.candidate_root = Path(candidate_root)
        self.calls = []
        self.metrics = {"history": 0, "model": 0, "candidate": 0, "state": 0}

    def send(self, frame):
        self.child.stdin.write(json.dumps(frame, ensure_ascii=False, allow_nan=False) + "\n")
        self.child.stdin.flush()

    def serve(self):
        for line in self.child.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                print("[host] 非法协议帧: %r" % line[:200], file=sys.stderr)
                return 2
            kind = frame.get("type")
            if kind == "history_request":
                self.metrics["history"] += 1
                self.send({"type": "history_response", "id": frame["id"], "records": []})
            elif kind == "model_request":
                self.metrics["model"] += 1
                call_id = "call-%d" % self.metrics["model"]
                self.calls.append(call_id)
                self.send({"type": "model_response", "id": frame["id"], "call_id": call_id,
                           "response": {"choices": [{"message": {"role": "assistant",
                                                                 "content": "train"}}],
                                        "usage": {"prompt_tokens": 1, "completion_tokens": 1}}})
            elif kind == "candidate_request":
                self.metrics["candidate"] += 1
                target = self.candidate_root / ("candidate-%d" % self.metrics["candidate"])
                target.mkdir(parents=True, exist_ok=True)
                for item in frame["files"]:
                    path = target / item["path"]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(base64.b64decode(item["data_base64"]))
                descriptor = frame["artifact"]
                manifest = [{"path": item["path"], "size_bytes": len(base64.b64decode(item["data_base64"])),
                             "sha256": "sha256:" + hashlib.sha256(base64.b64decode(item["data_base64"])).hexdigest()}
                            for item in frame["files"]]
                digest = "sha256:" + hashlib.sha256(json.dumps(
                    {"files": manifest, "artifact": descriptor}, allow_nan=False,
                    sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                self.send({"type": "candidate_response", "id": frame["id"],
                           "artifact_id": "artifact-candidate-%d" % self.metrics["candidate"],
                           "artifact_sha256": digest})
                self.last_candidate = target
            elif kind == "state_request":
                self.metrics["state"] += 1
                digest = "sha256:" + hashlib.sha256(b"state").hexdigest()
                self.send({"type": "state_response", "id": frame["id"], "state_sha256": digest})
            elif kind == "round_complete":
                record = frame.get("record", {})
                print("[host] round_complete: decision=%s epochs=%s selected=%s" % (
                    record.get("configuration", {}).get("decision"),
                    record.get("configuration", {}).get("epochs"),
                    record.get("selected_artifact_id")), file=sys.stderr)
                if not record.get("model_call_ids"):
                    print("[host] 缺少 model_call_ids", file=sys.stderr)
                    self.send({"type": "round_ack", "accepted": False})
                    return 3
                self.send({"type": "round_ack", "accepted": True})
                return 0
            elif kind == "round_failed":
                print("[host] round_failed: %s" % frame.get("error"), file=sys.stderr)
                return 4
            else:
                print("[host] 未知帧类型: %s" % kind, file=sys.stderr)
                return 5
        return 6


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "A"
    data_dir = Path(os.environ.get("MOCK_DATA", "/data"))
    previous = Path(os.environ.get("MOCK_PREVIOUS", "/previous"))
    work = Path(sys.argv[2] if len(sys.argv) > 2 else os.environ.get("MOCK_WORK", "/tmp/host"))
    work.mkdir(parents=True, exist_ok=True)
    candidate_root = work / "candidates"
    candidate_root.mkdir(exist_ok=True)
    entry = _entry()

    presets = {
        "A": ("classification", ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]),
        "B": ("regression", []),
        "C": ("regression", []),
    }
    kind, classes = presets[dataset]
    frame = _start_frame(dataset, str(data_dir / "train.jsonl"), str(data_dir / "validation.jsonl"),
                         str(previous), kind, classes, 512)

    env = dict(os.environ)
    env.pop("CONTEST_ARTIFACT_PATH", None)
    child = subprocess.Popen(entry + ["train"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=None, text=True, encoding="utf-8", env=env,
                             cwd=os.environ.get("MOCK_CWD", "/app"))
    host = Host(child, candidate_root)
    host.send(frame)
    code = host.serve()
    child.stdin.close()
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        child.kill()
    print("[host] train 结束: host=%d child=%d calls=%d" % (code, child.returncode, len(host.calls)),
          file=sys.stderr)
    if code != 0 or child.returncode != 0:
        return 1
    candidate = getattr(host, "last_candidate", None)
    if candidate is None:
        print("[host] 没有候选产物", file=sys.stderr)
        return 1
    checkpoint = candidate / "checkpoint.npz"
    if not checkpoint.is_file():
        print("[host] 候选缺少 checkpoint.npz", file=sys.stderr)
        return 1
    print("[host] 候选目录: %s (%d 字节)" % (candidate, checkpoint.stat().st_size), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
