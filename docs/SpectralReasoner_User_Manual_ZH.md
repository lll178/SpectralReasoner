# SpectralReasoner 中文说明书

## 1. 定位

SpectralReasoner 是一个轻量级、本地可部署的证据门控问答与受控生成引擎。主干路线是：

```text
Subword LM + SpectralReasoner + OSU Memory
```

它适合本地知识库问答、RAG 证据核验、拒答风险评估和轻量化私有部署。默认安装包不内置训练集、知识库、模型权重或运行输出。

## 2. 启动

先准备外部 bundle 和可选 JSONL 知识库，然后运行：

```powershell
pip install -e .

spectral-reasoner `
  --bundle C:\path\to\bundle `
  --kb C:\path\to\kb.jsonl `
  --host 0.0.0.0 `
  --port 8765
```

浏览器打开：

```text
http://127.0.0.1:8765/app
```

同一 Wi-Fi 下手机访问：

```text
http://YOUR_PC_LAN_IP:8765/app
```

## 3. 接口

### `/chat`

从请求传入的 `docs` 或启动时指定的本地知识库抽取证据片段，然后输出答案、拒答、风险、置信度和证据。

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

小型 subword LM 生成候选回复，谱推理层根据证据和风险重排。

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

## 4. 参数

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--bundle` | 必填 | 外部部署 bundle 路径。 |
| `--kb` | 空 | 外部 JSONL 知识库路径。 |
| `--host` | `127.0.0.1` | 监听地址；局域网访问用 `0.0.0.0`。 |
| `--port` | `8765` | HTTP 端口。 |
| `--device` | `auto` | `auto`、`cpu` 或 `cuda`。 |
| `--web-dir` | 包内前端 | 可替换为自定义前端目录。 |
| `--enable-infer` | 关闭 | 开启旧调试接口。 |

## 5. 输出字段

| 字段 | 含义 |
| --- | --- |
| `answer` | 最终答案；拒答时为 `null`。 |
| `refused` | 是否拒答。 |
| `risk` | 谱风险分数，越低越稳。 |
| `confidence` | 置信度。 |
| `evidence` | 支持答案的证据片段。 |
| `candidates` | 候选排序和诊断信息。 |
| `spectral_trace` | 谱熵、有效维数、条件数、相干性、LM 调用数和记忆轨迹。 |

## 6. 边界

SpectralReasoner 当前是工程原型，最适合证据支持型问答和受控生成。开放问答质量取决于外部知识库质量和文档切分方式。企业生产部署仍需要鉴权、日志、安全审计、监控和更大规模评测。
