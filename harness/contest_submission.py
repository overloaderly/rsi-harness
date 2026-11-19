import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from rsi_data import flatten
from rsi_model import Model

TASKS = ("classification", "regression")
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
CHECKPOINT = "checkpoint.npz"
EXPERIMENT = "experiment.json"
SEED = 42

KIND = {"A": "classification", "B": "regression", "C": "regression"}
CLASSES = {
    "A": ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"],
    "B": [],
    "C": [],
}
MAX_TRAIN_ROWS = {"A": 0, "B": 0, "C": 0}
MAX_VAL_ROWS = {"A": 0, "B": 0, "C": 0}
DEFAULT_EPOCHS = 3
MIN_BUDGET_SECONDS = 90.0
BUDGET_FRACTION = 0.72


def _log(message):
    print("[submission] %s" % (message,), file=sys.stderr, flush=True)


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_flag(name):
    return str(os.environ.get(name, "")).strip().lower() in ("1", "true", "yes", "on")


def _work():
    for candidate in (os.environ.get("CONTEST_WORK"), "/workspace", "/tmp"):
        if not candidate:
            continue
        root = Path(candidate)
        try:
            root.mkdir(parents=True, exist_ok=True)
            probe = root / (".write-probe-%d" % os.getpid())
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return root
        except OSError:
            continue
    return Path(tempfile.mkdtemp(prefix="harness-work-"))


def _rows(rows, task, classes, limit=0):
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
        elif isinstance(answer, (int, float)) and not isinstance(answer, bool) and np.isfinite(answer):
            labels.append(float(answer))
        else:
            labels.append(float("nan"))
            labeled = False
        if limit and len(ids) >= limit:
            break
    if not ids:
        raise ValueError("本轮数据为空")
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64 if task == "classification" else np.float64) if labeled else None
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
    if y_true is None:
        return None
    if task == "classification":
        f1 = _macro_f1(y_true, y_pred, len(classes))
        return {"score": round(f1, 6), "macro_f1": round(f1, 6),
                "accuracy": round(float(np.mean(np.asarray(y_pred) == y_true)), 6)}
    error = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    rmse = float(np.sqrt(np.mean(error ** 2)))
    clipped = np.clip(error, -50.0, 50.0)
    penalty = np.where(clipped < 0, np.expm1(np.clip(-clipped / 13.0, -30.0, 30.0)),
                       np.expm1(np.clip(clipped / 10.0, -30.0, 30.0)))
    nasa = float(np.sum(penalty))
    if not np.isfinite(nasa):
        nasa = 1e12
    return {"score": round(rmse, 6), "rmse": round(rmse, 6),
            "mae": round(float(np.mean(np.abs(error))), 6), "nasa": round(nasa, 6)}


def _evaluate(model, x, y, task, classes):
    if y is None:
        return None
    prediction = model.predict(x)
    if task == "classification":
        return _metrics(task, y, np.argmax(prediction, axis=1), classes)
    return _metrics(task, y, prediction, classes)


def _better(task, candidate, baseline):
    if not candidate:
        return False
    if not baseline:
        return True
    if task == "classification":
        return candidate["score"] > baseline["score"] + 1e-6
    return candidate["score"] < baseline["score"] - 1e-9


def _psi(reference, current, bins=10):
    reference = np.asarray(reference, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    if reference.size == 0 or current.size == 0:
        return 0.0
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        return 0.0
    edges = edges.copy()
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(reference, bins=edges)[0].astype(np.float64)
    cur_counts = np.histogram(current, bins=edges)[0].astype(np.float64)
    ref_ratio = np.clip(ref_counts / max(ref_counts.sum(), 1.0), 1e-6, None)
    cur_ratio = np.clip(cur_counts / max(cur_counts.sum(), 1.0), 1e-6, None)
    return float(np.sum((cur_ratio - ref_ratio) * np.log(cur_ratio / ref_ratio)))


def _drift(x_reference, x_current):
    count = min(x_reference.shape[1], x_current.shape[1])
    if count == 0:
        return {"psi_mean": 0.0, "psi_max": 0.0, "psi_frac_gt_0_25": 0.0}
    values = np.asarray([_psi(x_reference[:, i], x_current[:, i]) for i in range(count)], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"psi_mean": 0.0, "psi_max": 0.0, "psi_frac_gt_0_25": 0.0}
    return {"psi_mean": round(float(values.mean()), 6), "psi_max": round(float(values.max()), 6),
            "psi_frac_gt_0_25": round(float(np.mean(values > 0.25)), 6)}


def _kind(rnd, key):
    contract = rnd.task if isinstance(rnd.task, dict) else {}
    for field in ("kind", "task", "task_kind"):
        value = contract.get(field)
        if isinstance(value, str) and value in TASKS:
            return value
    return KIND.get(key, "classification")


def _classes(rnd, key, task):
    if task != "classification":
        return []
    contract = rnd.task if isinstance(rnd.task, dict) else {}
    for field in ("classes", "labels", "categories"):
        value = contract.get(field)
        if isinstance(value, list) and value and all(isinstance(item, str) and item for item in value):
            return list(value)
    return list(CLASSES.get(key, []))


def _parent_checkpoint(rnd):
    descriptor = getattr(getattr(rnd, "parent", None), "descriptor", {}) or {}
    declared = [item for item in (descriptor.get("checkpoint_paths") or []) if isinstance(item, str)]
    named = next((item for item in declared if item.lower().endswith(".npz")), declared[0] if declared else "")
    if named:
        try:
            return rnd.parent.path(named), named
        except Exception as error:
            code = getattr(error, "code", "")
            if code in ("PARENT_DIGEST_MISMATCH", "PARENT_PATH_UNSAFE", "PATH_UNSAFE"):
                raise
            _log("按声明读取父检查点失败(%s: %s)，改为扫描挂载根" % (code or type(error).__name__, error))
    root = Path(getattr(getattr(rnd, "parent", None), "root", "/previous"))
    for candidate in sorted(root.rglob("*.npz")):
        return candidate, candidate.name
    raise RuntimeError("找不到父检查点：/previous 下没有 .npz 文件")


def _history_summary(history):
    summary = []
    for record in (history or [])[-4:]:
        if not isinstance(record, dict):
            continue
        configuration = record.get("configuration") if isinstance(record.get("configuration"), dict) else {}
        summary.append({
            "id": record.get("id"),
            "hypothesis": str(record.get("hypothesis") or "")[:160],
            "validation": configuration.get("validation") if isinstance(configuration.get("validation"), dict) else None,
            "reason": str(record.get("reason") or "")[:160],
        })
    return summary


def _ask(rnd, payload):
    body = (
        "你是一次工业模型续训实验的调度器。下面是一轮实验的状态 JSON。"
        "请只输出一个英文单词：train（继续常规训练）、conserve（小步微调）、keep（本轮不训练，沿用父权重）。\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )
    receipt = rnd.model([{"role": "user", "content": body}],
                        max_tokens=min(48, rnd.max_output_tokens), temperature=0)
    text = (getattr(receipt, "text", "") or "").strip().lower()
    decision = "train"
    for keyword in ("keep", "conserve", "train"):
        if keyword in text:
            decision = keyword
            break
    return decision, receipt.call_id


def _plan_epochs(task, baseline, history, drift, base_epochs):
    epochs = base_epochs
    learning_rate_scale = 1.0
    last_score = None
    summary = _history_summary(history)
    if summary:
        validation = summary[-1].get("validation") or {}
        last_score = validation.get("score")
    if isinstance(last_score, (int, float)) and baseline is not None:
        if task == "classification":
            worse = last_score < baseline["score"] - 1e-6
        else:
            worse = last_score > baseline["score"] + 1e-9
        if worse:
            epochs, learning_rate_scale = max(1, base_epochs - 2), 0.5
        else:
            epochs = base_epochs + 1
    if drift["psi_mean"] < 0.10:
        epochs, learning_rate_scale = max(1, min(epochs, 1)), 0.5
    elif drift["psi_mean"] >= 0.25:
        epochs = epochs + 2
    return epochs, learning_rate_scale


def train(rnd):
    started = time.monotonic()
    budget = float(getattr(rnd, "wall_seconds", None) or 900.0)
    deadline = started + max(MIN_BUDGET_SECONDS, budget * BUDGET_FRACTION)
    key = str(rnd.dataset_key)
    task = _kind(rnd, key)
    classes = _classes(rnd, key, task)
    _log("round=%s dataset=%s task=%s budget=%.0fs" % (rnd.round_index, key, task, budget))

    history = rnd.history()
    _log("平台 history %d 条" % len(history))

    train_ids, x_train, y_train, paths = _rows(
        rnd.data.training_rows(), task, classes, limit=_env_int("RSI_MAX_TRAIN_ROWS", MAX_TRAIN_ROWS.get(key, 0)))
    val_ids, x_val, y_val, val_paths = _rows(
        rnd.data.validation_rows(), task, classes, limit=_env_int("RSI_MAX_VAL_ROWS", MAX_VAL_ROWS.get(key, 0)))
    if val_paths != paths:
        raise ValueError("训练集与验证集的特征结构不一致")
    if y_train is None:
        raise ValueError("训练集没有可用的标签")
    _log("train=%d rows, validation=%d rows, features=%d" % (len(train_ids), len(val_ids), len(paths)))

    parent_path, parent_name = _parent_checkpoint(rnd)
    model = Model.load(parent_path)
    if model.meta.get("task") != task:
        raise ValueError("父检查点任务类型为 %s，与 %s 不符" % (model.meta.get("task"), task))
    if list(model.meta["paths"]) != list(paths):
        raise ValueError("父检查点特征路径与公开数据不一致")
    if task == "classification" and list(model.meta.get("classes") or []) != list(classes):
        _log("提示：父检查点类别顺序与契约不同，沿用检查点顺序")
        classes = list(model.meta["classes"])

    baseline = _evaluate(model, x_val, y_val, task, classes)
    _log("父检查点基线: %s" % json.dumps(baseline, ensure_ascii=False))

    drift = _drift(x_train, x_val)
    _log("漂移: %s" % json.dumps(drift, ensure_ascii=False))

    base_epochs = max(1, _env_int("RSI_EPOCHS", DEFAULT_EPOCHS))
    planned_epochs, lr_scale = _plan_epochs(task, baseline, history, drift, base_epochs)
    learning_rate = float(model.meta["learning_rate"]) * lr_scale
    if time.monotonic() > deadline:
        planned_epochs = 0
        _log("时间预算不足，本轮跳过训练")

    decision, call_id = _ask(rnd, {
        "dataset_key": key,
        "task": task,
        "round_index": rnd.round_index,
        "parent_validation": baseline,
        "drift": drift,
        "planned_epochs": planned_epochs,
        "planned_learning_rate": learning_rate,
        "history": _history_summary(history),
    })
    epochs = planned_epochs
    if decision == "keep":
        epochs = 0
    elif decision == "conserve":
        epochs = max(0, min(planned_epochs, 2))
    _log("模型决策=%s call_id=%s epochs=%d lr=%.5f" % (decision, call_id, epochs, learning_rate))

    hypothesis = (
        "父检查点在公开 validation 上 %s=%s；漂移 psi_mean=%.3f；依据历史记录与一次平台模型决策(%s)选择 %d 个 epoch、学习率 %.5f。"
        % ("macro_f1" if task == "classification" else "rmse",
           baseline.get("score") if baseline else None, drift["psi_mean"], decision, epochs, learning_rate)
    )
    change_summary = "沿用父检查点标准化与类别映射，仅以公开训练集续训 NumPy MLP，第 %d 轮。" % rnd.round_index

    if epochs:
        model.meta["learning_rate"] = learning_rate
        model.fit(x_train, y_train, epochs)
    candidate_metrics = _evaluate(model, x_val, y_val, task, classes)
    improved = _better(task, candidate_metrics, baseline)

    work = _work()
    stamp = "%02d-%d" % (rnd.round_index, int(time.time()))
    candidate_dir = work / ("candidate-" + stamp)
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir, ignore_errors=True)
    candidate_dir.mkdir(parents=True, exist_ok=False)

    if epochs and not improved and baseline is not None:
        _log("候选变差(%s)，回滚到父权重" % json.dumps(candidate_metrics, ensure_ascii=False))
        model = Model.load(parent_path)
        selected_metrics = baseline
        reason = ("第 %d 轮续训后候选 %s 未优于父检查点 %s，按回滚策略提交父权重副本，保留实验记录供下一轮调整。"
                  % (rnd.round_index, candidate_metrics, baseline))
        hypothesis = "验证集续训无收益，本轮以回滚验证恢复能力。"
        change_summary = "不改变权重，仅复制父检查点作为候选，记录失败实验参数。"
    else:
        selected_metrics = candidate_metrics
        if candidate_metrics and baseline:
            reason = ("第 %d 轮：父 %s → 候选 %s（epochs=%d, lr=%.5f, 决策=%s），已选择本轮产物。"
                      % (rnd.round_index, json.dumps(baseline, ensure_ascii=False),
                         json.dumps(candidate_metrics, ensure_ascii=False), epochs, learning_rate, decision))
        elif epochs == 0:
            reason = "第 %d 轮：模型决策沿用父权重，提交父检查点副本。" % rnd.round_index
        else:
            reason = "第 %d 轮：验证集无标签，按既定步长提交第 %d 轮产物。" % (rnd.round_index, rnd.round_index)

    checkpoint_path = candidate_dir / CHECKPOINT
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    model.save(checkpoint_path)
    (candidate_dir / EXPERIMENT).write_text(json.dumps({
        "dataset_key": key, "task": task, "round": rnd.round_index,
        "parent_checkpoint": parent_name, "epochs": epochs, "learning_rate": learning_rate,
        "decision": decision, "seed": SEED, "drift": drift,
        "parent_validation": baseline, "candidate_validation": selected_metrics,
        "train_rows": len(train_ids), "validation_rows": len(val_ids),
    }, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    configuration = {
        "dataset_key": key, "task": task, "epochs": epochs,
        "learning_rate": round(float(learning_rate), 8), "decision": decision,
        "drift": drift, "validation": selected_metrics, "parent_validation": baseline,
        "train_rows": len(train_ids), "validation_rows": len(val_ids),
        "checkpoint_file": CHECKPOINT,
    }

    try:
        state_in = work / ("state-in-" + stamp)
        if state_in.exists():
            shutil.rmtree(state_in, ignore_errors=True)
        rnd.state_workspace(state_in)
        ledger = {"rounds": []}
        ledger_path = state_in / "ledger.json"
        if ledger_path.is_file():
            try:
                ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                ledger = {"rounds": []}
        if not isinstance(ledger, dict):
            ledger = {"rounds": []}
        ledger.setdefault("rounds", []).append({
            "round": rnd.round_index, "epochs": epochs, "learning_rate": learning_rate,
            "decision": decision, "drift": drift, "validation": selected_metrics,
            "parent_validation": baseline, "improved": bool(improved),
        })
        state_out = work / ("state-out-" + stamp)
        if state_out.exists():
            shutil.rmtree(state_out, ignore_errors=True)
        state_out.mkdir(parents=True, exist_ok=False)
        (state_out / "ledger.json").write_text(
            json.dumps(ledger, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        rnd.save_state(state_out)
    except Exception as error:
        _log("状态保存跳过（%s: %s）" % (type(error).__name__, error))

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
        history_ids=[record["id"] for record in history if isinstance(record, dict) and record.get("id")],
        model_call_ids=[call_id],
    )
    _log("round %s 完成，用时 %.1fs，决策 %s" % (rnd.round_index, time.monotonic() - started, decision))


_CACHE = {}


def _resolve_checkpoint(model_dir):
    chosen = None
    for name in ("CONTEST_ARTIFACT_PATH", "CONTEST_MODEL_FILE"):
        value = os.environ.get(name)
        if value:
            chosen = Path(value)
            break
    root = Path(model_dir)
    candidates = []
    if chosen is not None:
        candidates.append(chosen)
    candidates.append(root / CHECKPOINT)
    if root.is_dir():
        candidates.extend(sorted(root.rglob("*.npz")))
    for path in candidates:
        try:
            if path.is_file() and not path.is_symlink():
                return path
        except OSError:
            continue
    raise RuntimeError("在 %s 下找不到检查点（%s）" % (root, CHECKPOINT))


def _load(model_dir):
    path = _resolve_checkpoint(model_dir)
    key = str(path.resolve())
    model = _CACHE.get(key)
    if model is None:
        model = Model.load(path)
        _CACHE[key] = model
        _log("预测加载检查点 %s（epochs=%s）" % (path, model.meta.get("epochs")))
    return model


def _scores(probabilities):
    scores = [float(min(1.0, max(0.0, item))) for item in probabilities]
    if not scores:
        raise ValueError("模型没有输出任何类别分数")
    top = int(np.argmax(scores))
    for index in range(len(scores)):
        if index != top and scores[index] == scores[top]:
            scores[index] = max(0.0, scores[top] - 1e-6)
    total = float(sum(scores))
    if total > 1.0:
        scores = [value / total for value in scores]
    return scores


def predict(sample, model_dir):
    model = _load(model_dir)
    paths, values = flatten(sample.inputs)
    expected = list(model.meta["paths"])
    if paths != expected:
        raise ValueError("输入特征结构与检查点不一致（%d vs %d 项）" % (len(paths), len(expected)))
    x = np.asarray([values], dtype=np.float64)
    output = model.predict(x)
    if sample.task == "regression":
        value = float(np.asarray(output).reshape(-1)[0])
        if not np.isfinite(value) or value < 0:
            value = 0.0
        return {"outputs": {"value": value}}
    classes = list(model.meta.get("classes") or [])
    if not classes:
        raise ValueError("检查点缺少类别映射")
    scores = _scores(np.asarray(output).reshape(-1))
    return {"outputs": [{"label": label, "score": score} for label, score in zip(classes, scores)]}
