"""rsi-index 智训引擎 · Harness 工作区 v2 提交模块（基础模型路线）。

平台调用序列：
    python3 /app/harness.py describe   -> 声明任务类型
    python3 /app/harness.py train      -> 读 stdin 的 start 帧，跑一轮闭环
    python3 /app/harness.py predict    -> 逐行读样本，写预测

train 每轮做四件事（对应评分里的"读历史—做实验—依据结果决策"）：
    1. 读 /state 的历史记录 + 平台 history，搞清楚上一轮做了什么、结果如何
    2. 在公开 validation 上评估父检查点，得到本轮基线
    3. 续训出候选并重新评估：变好提交新权重，变差回滚到父权重（恢复能力）
    4. 用一次真实模型调用决定本轮步长，submit_candidate + finish 留下可核验记录

约定：stdout 只给协议 JSONL，日志一律写 stderr。
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from rsi_data import flatten
from rsi_model import Model

TASKS = ("classification", "regression")
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
CHECKPOINT = "checkpoint.npz"
SEED = 42

KIND = {"A": "classification", "B": "regression", "C": "regression"}
CLASSES = {
    "A": ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"],
    "B": [],
    "C": [],
}
# 公开数据规模：A 445 / B 7032 / C 113089 行；C 全量解析太慢，默认截断
MAX_TRAIN_ROWS = {"A": 5000, "B": 20000, "C": 40000}
MAX_VAL_ROWS = {"A": 5000, "B": 8000, "C": 8000}


def _log(message):
    print("[submission] " + str(message), file=sys.stderr, flush=True)


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _work():
    return Path(os.environ.get("CONTEST_WORK", "/workspace"))


# ------------------------------------------------------------------ 数据
def _rows(rows, task, classes, limit=0):
    """把平台 JSONL 行展开成 (ids, x, y|None, feature_paths)。"""
    ids, features, labels, paths = [], [], [], None
    labeled = True
    for row in rows:
        if not isinstance(row, dict) or "inputs" not in row:
            raise ValueError("数据行缺少 inputs")
        current, values = flatten(row["inputs"])
        if paths is None:
            paths = current
        elif current != paths:
            raise ValueError("本轮数据的特征结构与首行不一致")
        ids.append(str(row.get("id") or "row-%d" % len(ids)))
        features.append(values)
        answer = row.get("answer")
        if task == "classification":
            if isinstance(answer, str) and answer in classes:
                labels.append(classes.index(answer))
            else:
                labels.append(-1)
                labeled = False
        elif isinstance(answer, (int, float)) and not isinstance(answer, bool):
            labels.append(float(answer))
        else:
            labels.append(float("nan"))
            labeled = False
        if limit and len(ids) >= limit:
            break
    if not ids:
        raise ValueError("本轮数据为空")
    x = np.asarray(features, dtype=np.float64)
    y = None
    if labeled:
        y = np.asarray(labels, dtype=np.int64 if task == "classification" else np.float64)
    return ids, x, y, paths


def _macro_f1(y_true, y_pred, n_classes):
    scores = []
    for index in range(n_classes):
        tp = int(np.sum((y_pred == index) & (y_true == index)))
        fp = int(np.sum((y_pred == index) & (y_true != index)))
        fn = int(np.sum((y_pred != index) & (y_true == index)))
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def _metrics(task, y_true, y_pred, classes):
    if task == "classification":
        f1 = _macro_f1(y_true, y_pred, len(classes))
        return {"score": f1, "macro_f1": round(f1, 6),
                "accuracy": round(float(np.mean(y_pred == y_true)), 6)}
    error = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    rmse = float(np.sqrt(np.mean(error ** 2)))
    # NASA 非对称惩罚：提前预测罚得重、滞后预测罚得轻；指数要裁剪，否则大误差会溢出成 inf
    penalty = np.where(error < 0, np.expm1(np.clip(-error / 13.0, -30.0, 30.0)),
                       np.expm1(np.clip(error / 10.0, -30.0, 30.0)))
    nasa = float(np.sum(penalty))
    if not np.isfinite(nasa):
        nasa = 1e12
    return {"score": rmse, "rmse": round(rmse, 6), "mae": round(float(np.mean(np.abs(error))), 6),
            "nasa": round(nasa, 6)}


def _evaluate(model, x, y, task, classes):
    if y is None:
        return None
    prediction = model.predict(x)
    if task == "classification":
        return _metrics(task, y, np.argmax(prediction, axis=1), classes)
    return _metrics(task, y, prediction, classes)


def _better(task, candidate, baseline):
    if not candidate or not baseline:
        return bool(candidate)
    if task == "classification":
        return candidate["score"] > baseline["score"] + 1e-6
    return candidate["score"] < baseline["score"] - 1e-9


# ------------------------------------------------------------------ 契约读取
def _kind(rnd, key):
    contract = rnd.task if isinstance(rnd.task, dict) else {}
    kind = contract.get("kind") or contract.get("task") or KIND.get(key)
    if kind not in TASKS:
        raise ValueError("无法判定任务类型: %r" % (kind,))
    return kind


def _classes(rnd, key):
    contract = rnd.task if isinstance(rnd.task, dict) else {}
    for field in ("classes", "labels", "categories"):
        value = contract.get(field)
        if isinstance(value, list) and value and all(isinstance(item, str) and item for item in value):
            return list(value)
    return list(CLASSES.get(key, []))


def _parent_checkpoint(rnd):
    descriptor = getattr(rnd.parent, "descriptor", {}) or {}
    declared = [item for item in (descriptor.get("checkpoint_paths") or []) if isinstance(item, str)]
    named = next((item for item in declared if item.lower().endswith(".npz")), declared[0] if declared else "")
    if named:
        try:
            return rnd.parent.path(named), named
        except Exception as error:  # noqa: BLE001 - 退回扫描，仍然只读父目录
            _log("按声明读取父检查点失败(%s)，改为扫描 /previous" % error)
    for candidate in sorted(Path("/previous").rglob("*.npz")):
        return candidate, candidate.name
    raise RuntimeError("找不到父检查点：/previous 下没有 .npz 文件")


def _history_summary(history):
    summary = []
    for record in history[-4:]:
        configuration = record.get("configuration") if isinstance(record, dict) else None
        validation = configuration.get("validation") if isinstance(configuration, dict) else None
        summary.append({
            "id": record.get("id"),
            "hypothesis": str(record.get("hypothesis") or "")[:160],
            "validation": validation if isinstance(validation, dict) else None,
            "reason": str(record.get("reason") or "")[:160],
        })
    return summary


def _ask(rnd, payload):
    """让平台模型对本轮动作给一次真实决策；返回 (decision, call_id)。"""
    body = (
        "你是一次工业模型续训实验的调度器。下面是一轮实验的状态 JSON。"
        "请只输出一个英文单词：train（继续常规训练）、conserve（小步微调）、keep（本轮不训练，沿用父权重）。\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )
    receipt = rnd.model([{"role": "user", "content": body}],
                        max_tokens=min(48, rnd.max_output_tokens), temperature=0)
    text = (receipt.text or "").strip().lower()
    decision = "train"
    for keyword in ("keep", "conserve", "train"):
        if keyword in text:
            decision = keyword
            break
    return decision, receipt.call_id


# ------------------------------------------------------------------ 训练轮次
def train(rnd):
    started = time.monotonic()
    budget = float(rnd.wall_seconds or 900.0)
    deadline = started + max(90.0, budget * 0.75)
    key = str(rnd.dataset_key)
    task = _kind(rnd, key)
    classes = _classes(rnd, key)
    _log("round=%s dataset=%s task=%s budget=%.0fs" % (rnd.round_index, key, task, budget))

    history = rnd.history()
    _log("平台 history %d 条" % len(history))

    train_ids, x_train, y_train, paths = _rows(
        rnd.data.training_rows(), task, classes,
        limit=_env_int("RSI_MAX_TRAIN_ROWS", MAX_TRAIN_ROWS.get(key, 20000)))
    val_ids, x_val, y_val, val_paths = _rows(
        rnd.data.validation_rows(), task, classes,
        limit=_env_int("RSI_MAX_VAL_ROWS", MAX_VAL_ROWS.get(key, 8000)))
    if val_paths != paths:
        raise ValueError("训练集与验证集的特征结构不一致")
    _log("train=%d rows, validation=%d rows, features=%d" % (len(train_ids), len(val_ids), len(paths)))

    parent_path, parent_name = _parent_checkpoint(rnd)
    model = Model.load(parent_path)
    if model.meta["task"] != task:
        raise ValueError("父检查点任务类型为 %s，与 %s 不符" % (model.meta["task"], task))
    if list(model.meta["paths"]) != list(paths):
        raise ValueError("父检查点特征路径与公开数据不一致")
    if task == "classification" and list(model.meta["classes"]) != list(classes):
        _log("警告：父检查点类别顺序与平台契约不同，沿用检查点顺序")
        classes = list(model.meta["classes"])

    baseline = _evaluate(model, x_val, y_val, task, classes)
    _log("父检查点基线: %s" % json.dumps(baseline, ensure_ascii=False))

    base_epochs = max(1, _env_int("RSI_EPOCHS", 3))
    learning_rate = model.meta["learning_rate"]
    epochs = base_epochs
    if history and baseline:
        last = _history_summary(history)[-1].get("validation") or {}
        last_score = last.get("score")
        if isinstance(last_score, (int, float)):
            if task == "classification" and last_score < baseline.get("score", 0) - 1e-6:
                epochs, learning_rate = max(1, base_epochs - 2), learning_rate * 0.5
            elif task == "regression" and last_score > baseline.get("score", 0) + 1e-9:
                epochs, learning_rate = max(1, base_epochs - 2), learning_rate * 0.5
            else:
                epochs = base_epochs + 1
    if time.monotonic() > deadline:
        epochs = 0
        _log("时间预算不足，本轮跳过训练")

    decision, call_id = _ask(rnd, {
        "dataset_key": key,
        "task": task,
        "round_index": rnd.round_index,
        "parent_validation": baseline,
        "planned_epochs": epochs,
        "planned_learning_rate": learning_rate,
        "history": _history_summary(history),
    })
    _log("模型决策=%s call_id=%s" % (decision, call_id))
    if decision == "keep":
        epochs = 0
    elif decision == "conserve":
        epochs = max(1, min(epochs, 2))

    hypothesis = (
        "在公开 validation 上评估父检查点（%s=%.4f），依据历次记录与一次平台模型决策（%s）"
        "选择继续训练 %d 个 epoch、学习率 %.5f。"
        % ("macro_f1" if task == "classification" else "rmse",
           baseline.get("score", float("nan")) if baseline else float("nan"), decision, epochs, learning_rate)
    )
    change_summary = (
        "沿用父检查点的特征标准化与类别映射；仅以公开训练集续训 NumPy MLP，"
        "第 %d 轮，epochs=%d，lr=%.5f，seed=%d。"
        % (rnd.round_index, epochs, learning_rate, SEED)
    )

    if epochs:
        model.meta["learning_rate"] = learning_rate
        model.fit(x_train, y_train, epochs)
    candidate_metrics = _evaluate(model, x_val, y_val, task, classes)
    improved = _better(task, candidate_metrics, baseline)

    work = _work()
    stamp = "%02d-%d" % (rnd.round_index, int(time.time()))
    candidate_dir = work / ("candidate-" + stamp)
    candidate_dir.mkdir(parents=True, exist_ok=False)
    if not improved and epochs and baseline and candidate_metrics:
        _log("候选变差(%s)，回滚到父权重" % json.dumps(candidate_metrics, ensure_ascii=False))
        model = Model.load(parent_path)
        selected_metrics = baseline
        reason = ("第 %d 轮续训后候选 %s 未优于父检查点 %s，按回滚策略提交父权重副本；"
                  "保留本次实验记录供下一轮调整。" % (rnd.round_index, candidate_metrics, baseline))
        hypothesis = "验证集续训无收益，本轮以回滚验证“恢复”能力。"
        change_summary = "不改变权重，仅复制父检查点作为候选，记录失败实验参数。"
    else:
        selected_metrics = candidate_metrics
        if candidate_metrics and baseline:
            reason = ("第 %d 轮：父检查点 %s → 候选 %s（epochs=%d, lr=%.5f, 决策=%s），"
                      "已选择本轮产物。" % (rnd.round_index, json.dumps(baseline, ensure_ascii=False),
                                            json.dumps(candidate_metrics, ensure_ascii=False),
                                            epochs, learning_rate, decision))
        else:
            reason = "第 %d 轮：验证集无标签或为空，按既定步长提交第 %d 轮产物。" % (rnd.round_index, rnd.round_index)

    model.save(candidate_dir / CHECKPOINT)
    (candidate_dir / "experiment.json").write_text(json.dumps({
        "dataset_key": key, "task": task, "round": rnd.round_index,
        "parent_checkpoint": parent_name, "epochs": epochs, "learning_rate": learning_rate,
        "decision": decision, "seed": SEED,
        "parent_validation": baseline, "candidate_validation": selected_metrics,
        "train_rows": len(train_ids), "validation_rows": len(val_ids),
    }, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    configuration = {
        "dataset_key": key, "task": task, "epochs": epochs,
        "learning_rate": round(float(learning_rate), 8), "decision": decision,
        "validation": selected_metrics, "parent_validation": baseline,
        "train_rows": len(train_ids), "validation_rows": len(val_ids),
        "checkpoint_file": CHECKPOINT,
    }

    state_in = work / ("state-in-" + stamp)
    rnd.state_workspace(state_in)
    ledger = {"rounds": []}
    ledger_path = state_in / "ledger.json"
    if ledger_path.is_file():
        try:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            ledger = {"rounds": []}
    ledger.setdefault("rounds", []).append({
        "round": rnd.round_index, "epochs": epochs, "learning_rate": learning_rate,
        "decision": decision, "validation": selected_metrics, "parent_validation": baseline,
        "improved": bool(improved),
    })
    state_out = work / ("state-out-" + stamp)
    state_out.mkdir(parents=True, exist_ok=False)
    (state_out / "ledger.json").write_text(
        json.dumps(ledger, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    rnd.save_state(state_out)

    submitted = rnd.submit_candidate(
        candidate_dir,
        hypothesis=hypothesis,
        change_summary=change_summary,
        configuration=configuration,
        seed=SEED,
        checkpoint_paths=[CHECKPOINT],
        code_paths=[],
        predict_argv=PREDICT_ARGV,
    )
    rnd.finish(
        selected_artifact_id=submitted["artifact_id"],
        hypothesis=hypothesis,
        change_summary=change_summary,
        reason=reason,
        configuration=configuration,
        seed=SEED,
        history_ids=[record["id"] for record in history],
        model_call_ids=[call_id],
    )
    _log("round %s 完成，用时 %.1fs" % (rnd.round_index, time.monotonic() - started))


# ------------------------------------------------------------------ 推理
_CACHE = {}


def _load(model_dir):
    key = str(model_dir)
    if key in _CACHE:
        return _CACHE[key]
    root = Path(model_dir)
    chosen = os.environ.get("CONTEST_MODEL_FILE")
    candidates = []
    if chosen:
        candidates.append(Path(chosen))
    candidates.append(root / CHECKPOINT)
    if root.is_dir():
        candidates.extend(sorted(root.rglob("*.npz")))
    for path in candidates:
        if path.is_file():
            model = Model.load(path)
            _CACHE[key] = model
            _log("预测加载检查点 %s（epochs=%s）" % (path, model.meta.get("epochs")))
            return model
    raise RuntimeError("在 %s 下找不到检查点（%s）" % (root, CHECKPOINT))


def predict(sample, model_dir):
    model = _load(model_dir)
    paths, values = flatten(sample.inputs)
    if paths != list(model.meta["paths"]):
        raise ValueError("输入特征结构与检查点不一致（%d vs %d 项）" % (len(paths), len(model.meta["paths"])))
    x = np.asarray([values], dtype=np.float64)
    if sample.task == "regression":
        value = float(model.predict(x)[0])
        return {"value": max(0.0, value) if np.isfinite(value) else 0.0}
    probabilities = model.predict(x)[0]
    classes = list(model.meta["classes"])
    scores = [float(min(1.0, max(0.0, item))) for item in probabilities]
    top = int(np.argmax(scores))
    if scores.count(scores[top]) > 1:
        scores[top] = min(1.0, scores[top] + 1e-9)
    return [{"label": label, "score": score} for label, score in zip(classes, scores)]
