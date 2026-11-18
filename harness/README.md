# rsi-index Harness 镜像（智训引擎 · 基础模型路线）

官方 SDK：`contest-harness-sdk-1.0.2.tar.gz`（SHA-256 `c676c8be…f85133`），
下载地址 `https://rsi-index.115.zone/downloads/contest-harness-sdk-1.0.2.tar.gz`。
本目录已经把 SDK 源码 vendored 进 `vendor/`，不依赖外部下载即可构建。

## 目录

| 路径 | 作用 |
| --- | --- |
| `contest_submission.py` | 选手模块：`train(rnd)` 跑一轮闭环，`predict(sample, model_dir)` 逐行推理 |
| `rsi_model.py` | 参考模型实现（NumPy MLP + Adam），与官方 `contest.cpu-mlp.v1` 检查点格式完全一致 |
| `rsi_data.py` | 输入展开（按字典键排序、列表下标编号），保证特征路径与父检查点一致 |
| `harness.py` | 官方入口模板，7 行，平台以 `describe/train/predict` 调用 |
| `vendor/` | SDK 1.0.2 源码（`pip install` 用） |
| `selftest.py` | 离线自测：假宿主跑 N 轮 train + 一次 predict，不需要平台；`--demo` 时输出可直接录屏的分轮讲解 |
| `demo.bat` | 一键跑「三轮训练 + predict」演示（录屏用，`demo.bat A pause` 可逐轮暂停） |
| `Dockerfile` | 镜像定义 |
| `build.bat` | Windows 一键构建 + 自测 + `docker save` |
| `.github/workflows/build-harness.yml` | 云端构建（无需本地 Docker） |

## 构建与导出

```bat
build.bat
```

或手动：

```bash
docker build --platform linux/amd64 -t rsi-harness:v1 .
docker run --rm rsi-harness:v1 describe
docker save rsi-harness:v1 -o rsi-harness-v1.tar
```

国内网络下 pip 可能连不上 PyPI，加一个镜像源参数即可（Dockerfile 里已留 `ARG PIP_INDEX_URL`）：

```bash
docker build --platform linux/amd64 \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  -t rsi-harness:v1 .
```

> 注意：镜像是带 ENTRYPOINT 的，`docker run rsi-harness:v1 describe` 只能接子命令，不要再写 `python3 /app/harness.py`，否则会被当成多余参数。

`describe` 期望输出：

```json
{"protocol":"contest.harness.workspace.v2","model_access":"host_gateway_rpc","sdk":"contest_harness","sdk_version":"1.0.2","tasks":["classification","regression"]}
```

## 上传

1. 站点 → 作品提交 → Harness 镜像 → 上传 `rsi-harness-v1.tar`（**未压缩**单镜像 tar，linux/amd64）。
2. 等状态从"发布与核验中"变成"已就绪"，可以离页，**不要重复上传**。
3. `+ 新建提交版本`：选该镜像 + 作品文档（≤2 MiB）+ 模型身份（基础模型：白名单 + 二次确认）
   + 训练记录链接（https，写明未使用微调权重）+ 源码仓库链接 → 核对并保存新版本 → 校验 → 试运行。
4. 归档保留期结束后镜像会失效，重新 `docker save` 上传即可。

## 离线自测

```bash
# 合成数据，任务 A
docker run --rm --user 0 --entrypoint python3 rsi-harness:v1 /opt/harness-selftest/selftest.py --task A
# 真实公开数据（挂载 labs）
docker run --rm --user 0 --entrypoint python3 -v B:\whale\智训引擎\labs:/labs:ro rsi-harness:v1 \
  /opt/harness-selftest/selftest.py --task B --data-dir /labs/cpu-b/public
```

`--user 0` 是因为自测要写 `/previous` 和 `/model`（平台运行时这两个目录由平台挂载）。

## 演示（录屏用）

```bat
demo.bat               :: 任务 A，三轮，一口气跑完
demo.bat A pause       :: 每轮结束等回车，边讲边录
demo.bat C             :: 任务 C（发动机 RUL），两轮
```

演示模式用 `labs/` 里的**官方父检查点**当基线、用整份公开验证集评分，所以数字可复现（任务 A：0.7140 → 0.7187 → 0.7217，第三轮 `conserve` 后变差并**自动回滚**）。分镜与台词见 `../演示视频_分镜与台词.md`。

等价的完整命令：

```bash
docker run --rm --user 0 --entrypoint python3 -v <labs>:/labs:ro rsi-harness:v1 \
  /opt/harness-selftest/selftest.py --task A --rounds 3 --demo \
  --data-dir /labs/cpu-a/public --parent /labs/cpu-a/checkpoint.npz --train-rows 500
```

## 每轮 train 做了什么

1. 读平台 `history` 与 `/state` 里的 `ledger.json`，得到历次实验和指标。
2. 在公开 `validation` 上评估父检查点，作为本轮基线（分类 Macro-F1 / 回归 RMSE）。
3. 依据历史决定步长，再用一次真实模型调用（`rnd.model`）决定 `train / conserve / keep`。
4. 续训候选并复评：优于基线 → 提交新权重；不如基线 → **回滚**，提交父权重副本并记录失败实验。
5. `submit_candidate` + `finish` 写清 hypothesis / change_summary / reason / 指标，供专家评分。

## 可调环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `RSI_EPOCHS` | 3 | 每轮基础训练轮数 |
| `RSI_MAX_TRAIN_ROWS` | A 5000 / B 20000 / C 40000 | 训练集截断（C 公开 11.3 万行，全量解析太慢） |
| `RSI_MAX_VAL_ROWS` | A 5000 / B 8000 / C 8000 | 验证集截断 |
| `CONTEST_SUBMISSION` | `contest_submission` | 选手模块名（SDK 自带） |
| `CONTEST_WORK` | `/workspace` | 可写工作目录 |

## 常见坑

- 打包成 ZIP / 源码 tar / gzip 一律被拒，只能 `docker save` 的未压缩 tar。
- 必须 `--platform linux/amd64`，Apple Silicon / ARM 上构建会因架构不符被拒。
- 镜像里不要写死任何 API Key；模型经平台网关 `rnd.model()` / `rnd.transport()` 调用。
- 训练日志只能走 stderr，stdout 是协议通道。
- 候选目录 ≤64 文件 / 64 MiB；检查点约 130 KB，没问题。
