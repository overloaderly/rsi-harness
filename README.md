# 智训引擎 · Harness 镜像源码

「大学生"人工智能+"创新大赛」智训引擎企业赛的选手镜像源码。镜像实现平台 `contest.harness.workspace.v2` 协议，
在官方父检查点基础上做**续训 + 评估 + 回滚**，LLM 只通过平台网关调用（基础模型路线，**不使用任何微调权重**）。

## 目录

```
harness/                     镜像构建目录（docker build 的上下文）
├── Dockerfile               镜像定义：python3.12-slim + contest_harness 1.0.2 + numpy
├── harness.py               平台入口（平台追加 describe / train / predict）
├── contest_submission.py    选手模块：train(rnd) / predict(sample, model_dir)
├── rsi_model.py             CPU 友好的 MLP（Adam、种子可复现、npz 检查点）
├── rsi_data.py              特征展开（与官方父检查点 paths 完全对齐）
├── selftest.py              离线端到端假宿主，可脱离平台验证整条链路
├── vendor/                  contest_harness SDK 1.0.2（随镜像安装）
├── build.bat                一键构建 + 自测 + 导出 tar（Windows）
└── .github/workflows/       GitHub Actions 构建产物（免本地 Docker）
docs/
└── 训练记录_基础模型版.md    实验记录（模型身份声明、≥3 轮闭环、失败与回滚）
```

## 构建

```bash
cd harness
docker build --platform linux/amd64 -t rsi-harness:v1 .
# 国内网络拉 PyPI 慢时：
docker build --platform linux/amd64 \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  -t rsi-harness:v1 .
```

镜像带 ENTRYPOINT，自检时只接子命令：

```bash
docker run --rm rsi-harness:v1 describe          # 输出协议 JSON
docker run --rm --entrypoint python3 rsi-harness:v1 /opt/harness-selftest/selftest.py --task A
```

导出比赛要求的**未压缩 tar**：

```bash
docker save rsi-harness:v1 -o rsi-harness-v1.tar
```

## 每轮 train 做什么

1. 读 `rnd.history()` 与 `/state` 的 ledger，决定本轮 epochs / 学习率；
2. 载入 `/previous` 的父检查点，在 `/data/train.jsonl` 上评估基线；
3. 调一次 `rnd.model()`，让平台 LLM 给出 `train` / `conserve` 决策（异常时按本地规则兜底）；
4. 续训 → 在验证集复评 → 优于父检查点才采纳，否则回滚提交父权重副本；
5. `submit_candidate()` 提交 `checkpoint.npz` + `experiment.json`，`finish()` 带上 history_ids 与 model_call_ids。

## 已验证

- 容器内离线自测 A/B/C 全部通过（A 走"提升→采纳"，B/C 正确触发"变差→回滚"）；
- 真实公开数据挂载跑通，父检查点特征对齐、六类标签顺序一致；
- `docker save` 归档可被 `docker load` 原样回灌（linux/amd64）。

## 协议合规自检（v3）

镜像遵循官方 `contest.harness.workspace.v2` 协议，三条入口命令均在镜像内实测通过：

| 命令 | 作用 | 实测结果 |
|---|---|---|
| `python3 /app/harness.py describe` | 自描述入口声明 | `{"protocol":"contest.harness.workspace.v2","sdk_version":"1.0.2","tasks":["classification","regression"]}` |
| `... train` | 一轮训练（读 /data 与 /previous，写候选、state，finish） | B 任务 7032 行 / 34 维，父基线 RMSE 681.16，PSI 2.28，调模型一次，产出 candidate 36604 字节，round_complete accepted |
| `... predict` | 逐行预测 | A/B/C 三个数据集在容器内分别输出 6 分类概率与回归值，格式经官方 `normalize_output` 校验 |

四轮自查发现并修复的具体问题：

1. **协议层**：旧实现只读 `CONTEST_MODEL_FILE`，而协议下发的是 `CONTEST_ARTIFACT_PATH`；分类输出裸列表、并列最高分会越界。已全部修正。
2. **训练层**：`except Exception` 过宽会把 `PARENT_DIGEST_MISMATCH` 吞掉并降级为目录扫描，等于绕过摘要校验；已改为对协议错误码直接抛出。
3. **预测层**：用官方 `normalize_output` 逐行校验，A 1244/1244、B 1679/1679、C 3000/3000 全部通过；C 全量 23049 行耗时 8.7 s。
4. **工具层**：离线自检脚本原先硬编码 `/tmp`、`/app`、`/previous`、`/model`，已改为可配置常量；SDK 自身固定 `/previous` 与 `/model`，故完整端到端只在镜像内验证。

镜像规格：`linux/amd64`，`docker save` 未压缩 tar **185.7 MB**（平台上限 2 GB）。
