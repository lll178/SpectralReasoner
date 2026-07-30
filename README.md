# SpectralReasoner

中文 | [English](#english)

SpectralReasoner 是一个轻量级、本地可部署的证据门控问答与受控生成引擎。

主干路线：

```text
Subword LM + SpectralReasoner + OSU Memory
```

它适合做本地知识库问答、RAG 证据核验、低成本受控生成、拒答风险评估和轻量化私有部署。默认发布包只包含源码、文档、前端和许可证；不包含训练集、示例数据、模型权重、运行结果或额外研究材料。

## 核心优势

- 本地优先：默认不依赖外部 API。
- 低成本推理：主干路径优先使用少量 LM 前向调用，再由谱风险层重排。
- 证据门控：输出包含答案、证据、风险、置信度和候选排序。
- 可拒答：缺证据、高冲突或高风险时可返回 refused。
- OSU Memory：可记录 unknown、conflict、high-risk 谱态，用于后续召回和风险先验。
- 轻量部署：Python 标准库 HTTP server + PyTorch + NumPy，可在局域网内给手机浏览器访问。

## 安装

```powershell
pip install -e .
```

或只安装依赖：

```powershell
pip install -r requirements.txt
```

## 准备运行资源

发布包是纯安装包，不内置模型或数据。你需要在本地准备：

- `--bundle`：训练或转换得到的 SpectralReasoner 部署 bundle，通常包含 `model.pt`、`vocab.json`、`lm_config.json`、`reasoner_config.json`。
- `--kb`：可选 JSONL 本地知识库。每行一个 JSON 对象，建议包含 `text` 字段。

注意：CMRC2018 生成的 KB 主要用于阅读理解、证据恢复和工程测试，不是干净的通用中文常识库。真实部署时应换成你自己的领域 KB，并保证文档来源、切分方式和问题分布与业务场景一致。

## 启动服务

```powershell
spectral-reasoner `
  --bundle C:\path\to\bundle `
  --kb C:\path\to\kb.jsonl `
  --host 0.0.0.0 `
  --port 8765
```

打开前端：

```text
http://127.0.0.1:8765/app
```

同一 Wi-Fi 下手机访问：

```text
http://YOUR_PC_LAN_IP:8765/app
```

## API

### `/chat`

从 `docs` 或本地 `--kb` 自动抽取候选 evidence span，再由 SpectralReasoner 重排并决定回答或拒答。

```json
{
  "messages": [
    {"role": "user", "content": "中国在哪里？"}
  ],
  "docs": [
    "中华人民共和国位于亚洲东部、太平洋西岸，首都是北京。"
  ],
  "max_candidates": 8,
  "kind": "known"
}
```

### `/generate-chat`

小型 subword LM 先生成多个候选回复，再由谱推理层根据证据、风险和相干性重排。

```json
{
  "messages": [
    {"role": "user", "content": "香港在哪里？请用一句话回答。"}
  ],
  "docs": [
    "香港是中华人民共和国特别行政区，位于中国南部、珠江口以东，北接广东省深圳市。"
  ],
  "generated_candidates": 6,
  "max_new_tokens": 48,
  "temperature": 0.9,
  "top_k": 40,
  "kind": "known"
}
```

### Legacy Debug API

`/infer` 和 `/batch` 默认关闭，仅用于调试或 benchmark。

```powershell
spectral-reasoner --bundle C:\path\to\bundle --enable-infer
```

## 主要参数

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--bundle` | 必填 | 外部部署 bundle 路径。 |
| `--kb` | 空 | 外部 JSONL 本地知识库。 |
| `--host` | `127.0.0.1` | 监听地址；手机访问建议 `0.0.0.0`。 |
| `--port` | `8765` | HTTP 端口。 |
| `--device` | `auto` | `auto`、`cpu` 或 `cuda`。 |
| `--web-dir` | 包内前端 | 可替换为自定义前端目录。 |
| `--enable-infer` | 关闭 | 开启旧 `/infer` 和 `/batch` 调试接口。 |

## 输出字段

| 字段 | 含义 |
| --- | --- |
| `answer` | 最终答案；拒答时为 `null`。 |
| `refused` | 是否拒答。 |
| `risk` | 风险分数，越低越稳。 |
| `confidence` | 置信度。 |
| `evidence` | 支持答案的证据片段。 |
| `route` | 推理路径，如 `mainline` 或 `generate_chat_mainline`。 |
| `candidates` | 候选答案排序和诊断信息。 |
| `spectral_trace` | 谱熵、有效维数、条件数、相干性、LM 调用数和记忆轨迹。 |

## 当前评测参考

在内部 CMRC2018 证据恢复评测中，主干路径达到：

| 模式 | Top-1 Accuracy | Recovery | Mean LM Forwards |
| --- | ---: | ---: | ---: |
| clean | 1.0000 | 0.0000 | 1.00 |
| missing | 1.0000 | 1.0000 | 1.00 |
| distractor | 0.9688 | 0.6562 | 1.00 |
| overall | 0.9896 | 0.5521 | 1.00 |

这些数字来自受控阅读理解和证据恢复任务，不代表开放互联网问答准确率。实际效果取决于本地知识库质量、文档切分方式和问题分布。

## 模型规模与最小部署建议

### 参考小型 Bundle

当前参考 bundle 是小型本地 subword LM 加 SpectralReasoner 运行层。

| 项目 | 数值 |
| --- | ---: |
| 参数量 | 约 1.03M |
| 模型权重大小 | 约 4 MB |
| 干净源码包 | 约 0.46 MB |
| 干净源码 zip | 约 0.12 MB |
| 运行时 | Python + PyTorch + NumPy |
| 用户主入口 | `/chat`、`/generate-chat` |

安装包本身不包含模型权重或训练数据。实际运行体积取决于你训练或接入的外部 bundle。

### 最小部署建议

| 目标 | 建议最低配置 |
| --- | --- |
| CPU | 2 核或以上 |
| RAM | 1 GB 可用内存 |
| 磁盘 | 50 MB 用于源码、依赖配置和小型 bundle |
| Python | 推荐 3.10+ |
| 运行时 | PyTorch CPU build + NumPy |
| GPU | 非必需 |
| 网络 | 本地资源准备完成后不需要联网 |

本安装包不提供训练集。请按需自行下载合法数据集，并根据说明书或训练入口生成本地 bundle。手机端不直接训练模型，建议先在 PC 端完成训练或 bundle 准备，再通过局域网 API 或自定义移动端前端接入。

## 发布包边界

默认发布包不包含：

- 训练集或知识库数据。
- 模型权重和部署 bundle。
- 运行目录、缓存、日志和 benchmark 输出。
- 历史实验脚本目录。

## 许可证

本项目采用双轨制：

```text
Non-commercial source-available license + paid commercial license
```

任何商业使用必须取得 Huang Hansong 的书面商业授权。未取得单独商业授权前，基于本软件的修改版、衍生版、fork、插件、封装、集成、服务端修改或托管部署必须继续使用本许可证，并公开完整对应源代码。

详见：

- `LICENSE`
- `COMMERCIAL_LICENSE.md`
- `THIRD_PARTY_NOTICES.md`

商业合作、企业部署、定制训练或技术支持，请通过 GitHub 仓库 issue/discussion 或作者 GitHub 主页列出的商务联系方式联系作者。

## 边界

SpectralReasoner 当前是工程原型。它最适合证据支持型问答和受控生成，不是无限制开放聊天大模型。企业生产部署仍需要鉴权、日志、安全审计、监控和更大规模评测。

CMRC2018 KB 是测试知识库，不应被当成默认通用知识库。真实部署需要接入干净、可控、可授权的领域知识库。

---

<a id="english"></a>

# SpectralReasoner

SpectralReasoner is a lightweight, local, evidence-gated QA and controlled generation engine.

Main route:

```text
Subword LM + SpectralReasoner + OSU Memory
```

It is designed for local knowledge-base QA, RAG evidence checking, low-cost controlled generation, refusal/risk scoring, and private lightweight deployment. The default release package contains only source code, documentation, frontend assets, and license files. It does not include datasets, example data, model weights, run outputs, or extra research materials.

## Highlights

- Local-first, with no external API required by default.
- Low-cost inference with a small LM prior and spectral risk reranking.
- Evidence-gated output with answer, evidence, risk, confidence, and candidate ranking.
- Refusal support for missing evidence, conflict, or high-risk states.
- OSU Memory hooks for unknown, conflict, and high-risk spectral states.
- Lightweight deployment with Python stdlib HTTP server, PyTorch, and NumPy.

## Install

```powershell
pip install -e .
```

Or install dependencies only:

```powershell
pip install -r requirements.txt
```

## Runtime Assets

The release package is clean and does not bundle models or data. You need to provide:

- `--bundle`: an external SpectralReasoner deployment bundle containing files such as `model.pt`, `vocab.json`, `lm_config.json`, and `reasoner_config.json`.
- `--kb`: an optional external JSONL knowledge base. Each row should contain a `text` field.

Note: a KB generated from CMRC2018 is mainly for reading-comprehension,
evidence-recovery, and engineering tests. It is not a clean general-purpose
Chinese commonsense knowledge base. Real deployments should use a domain KB
whose source, chunking strategy, and question distribution match the target
workflow.

## Run

```powershell
spectral-reasoner `
  --bundle C:\path\to\bundle `
  --kb C:\path\to\kb.jsonl `
  --host 0.0.0.0 `
  --port 8765
```

Open:

```text
http://127.0.0.1:8765/app
```

Phone on the same Wi-Fi:

```text
http://YOUR_PC_LAN_IP:8765/app
```

## API

### `/chat`

Extract candidate evidence spans from request docs or the local KB, then rerank and decide whether to answer or refuse.

```json
{
  "messages": [
    {"role": "user", "content": "Where is China?"}
  ],
  "docs": [
    "The People's Republic of China is located in East Asia, on the western Pacific coast. Its capital is Beijing."
  ],
  "max_candidates": 8,
  "kind": "known"
}
```

### `/generate-chat`

A small subword LM generates candidate replies; the spectral layer reranks them by evidence, risk, and coherence.

```json
{
  "messages": [
    {"role": "user", "content": "Where is Hong Kong? Answer in one sentence."}
  ],
  "docs": [
    "Hong Kong is a special administrative region of China, located in southern China east of the Pearl River estuary."
  ],
  "generated_candidates": 6,
  "max_new_tokens": 48,
  "temperature": 0.9,
  "top_k": 40,
  "kind": "known"
}
```

## Evaluation

CMRC2018 controlled evidence-recovery benchmark:

| Mode | Top-1 Accuracy | Recovery | Mean LM Forwards |
| --- | ---: | ---: | ---: |
| clean | 1.0000 | 0.0000 | 1.00 |
| missing | 1.0000 | 1.0000 | 1.00 |
| distractor | 0.9688 | 0.6562 | 1.00 |
| overall | 0.9896 | 0.5521 | 1.00 |

These results are from controlled reading-comprehension and evidence-recovery
tasks. They are not open-web QA accuracy claims.

## Model Size and Minimal Deployment

### Reference Small Bundle

The current reference bundle is a small local subword LM plus the
SpectralReasoner runtime layer.

| Item | Value |
| --- | ---: |
| Parameters | about 1.03M |
| Model weight size | about 4 MB |
| Clean source package | about 0.46 MB |
| Clean source zip | about 0.12 MB |
| Runtime | Python + PyTorch + NumPy |
| Main user endpoints | `/chat`, `/generate-chat` |

The source package does not include model weights or training data. Runtime
bundle size depends on the external bundle you train or attach.

### Minimal Deployment Recommendation

| Target | Recommended Minimum |
| --- | --- |
| CPU | 2 cores or above |
| RAM | 1 GB free memory |
| Disk | 50 MB for source, configuration, and a small bundle |
| Python | 3.10+ recommended |
| Runtime | PyTorch CPU build + NumPy |
| GPU | Not required |
| Network | Not required after local assets are prepared |

This package does not provide training datasets. Download legal datasets as
needed and build a local bundle according to the manual or training entrypoints.
Phone-side deployment should use a model trained or prepared on a PC first, then
connect through the local API or a custom mobile frontend.

## License

SpectralReasoner uses a dual-track model:

```text
Non-commercial source-available license + paid commercial license
```

Commercial use requires a separate written commercial license from Huang Hansong. Unless separately licensed, derivative works, forks, wrappers, plugins, integrations, service-side modifications, and hosted deployments must remain under this license and disclose complete corresponding source code.

See `LICENSE`, `COMMERCIAL_LICENSE.md`, and `THIRD_PARTY_NOTICES.md`.

## Boundary

A CMRC2018-derived KB is a test KB, not a default general knowledge base. Real
deployments should attach a clean, controlled, and properly licensed domain
knowledge base.
