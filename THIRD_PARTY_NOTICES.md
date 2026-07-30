# Third-Party Notices / 第三方依赖声明

This is an engineering-level dependency notice for the current
SpectralReasoner package. It is not a substitute for legal review.

这是当前 SpectralReasoner 安装包的工程级依赖声明，不替代正式法律审查。

## Runtime Dependencies

| Dependency | Use | Typical License |
| --- | --- | --- |
| Python | Runtime | Python Software Foundation License |
| PyTorch | Small local LM runtime/training | BSD-style |
| NumPy | Spectral/Hankel numeric computation | BSD |

The HTTP server uses Python standard library modules.

服务端 HTTP 层使用 Python 标准库实现。

## External Assets

SpectralReasoner does not grant rights to external datasets, local knowledge
bases, trained weights, or model bundles. If you train or redistribute any
runtime asset, you are responsible for checking the upstream terms.

SpectralReasoner 不授予外部数据集、本地知识库、训练权重或模型 bundle 的权利。训练或再分发任何运行资源时，使用者需要自行确认上游条款。

## Copyleft Audit Note

The current product path is designed to avoid GPL/AGPL/LGPL runtime
dependencies. Before public release or commercial deployment, run your own
dependency audit on the exact packaged artifact:

```powershell
rg "GPL|AGPL|LGPL|Apache|MIT|BSD|license" -S .
pip-licenses
```

## Contributor Rule

Do not contribute copied code from GPL, AGPL, LGPL, or other copyleft projects
unless the project owner explicitly approves it in writing.

未经项目所有者书面批准，不要提交从 GPL、AGPL、LGPL 或其他 copyleft 项目复制的代码。
