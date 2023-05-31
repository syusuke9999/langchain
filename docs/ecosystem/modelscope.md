# ModelScope（モデルスコープ）

このページでは、LangChain内でModelScopeエコシステムを使用する方法について説明します。
インストールとセットアップ、そして特定のModelScopeラッパーへの参照の2つの部分に分かれています。

## インストールとセットアップ

* Python SDKを`pip install modelscope`でインストールします。

## ラッパー

### 埋め込み

Modelscopeの埋め込みラッパーが存在し、以下のようにアクセスできます。

```python
from langchain.embeddings import ModelScopeEmbeddings
```

これに関する詳細なウォークスルーは、
[このノートブック](../modules/models/text_embedding/examples/modelscope_hub.ipynb)を参照してください。
