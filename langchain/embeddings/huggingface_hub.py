"""Wrapper around HuggingFace Hub embedding models."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Extra, root_validator

from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_dict_or_env

DEFAULT_REPO_ID = "sentence-transformers/all-mpnet-base-v2"
VALID_TASKS = ("feature-extraction",)


class HuggingFaceHubEmbeddings(BaseModel, Embeddings):
    """HuggingFaceHub埋め込みモデルをラップするクラス。

        このクラスを使用するには、``huggingface_hub`` Pythonパッケージがインストールされていることと、
        環境変数``HUGGINGFACEHUB_API_TOKEN``にAPIトークンが設定されているか、
        またはコンストラクタに名前付きパラメータとして渡す必要があります。

        例：
            .. code-block:: python

                from langchain.embeddings import HuggingFaceHubEmbeddings
                repo_id = "sentence-transformers/all-mpnet-base-v2"
                hf = HuggingFaceHubEmbeddings(
                    repo_id=repo_id,
                    task="feature-extraction",
                    huggingfacehub_api_token="my-api-key",
                )
        """

    client: Any  #: :meta private:
    repo_id: str = DEFAULT_REPO_ID
    """使用するモデル名"""
    task: Optional[str] = "feature-extraction"
    """モデルを呼び出すタスク"""
    model_kwargs: Optional[dict] = None
    """モデルに渡すキーワード引数。"""

    huggingfacehub_api_token: Optional[str] = None

    class Config:
        """このpydanticオブジェクトの設定。"""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """環境にapiキーとpythonパッケージが存在することを確認する。"""
        huggingfacehub_api_token = get_from_dict_or_env(
            values, "huggingfacehub_api_token", "HUGGINGFACEHUB_API_TOKEN"
        )
        try:
            from huggingface_hub.inference_api import InferenceApi

            repo_id = values["repo_id"]
            if not repo_id.startswith("sentence-transformers"):
                raise ValueError(
                    "Currently only 'sentence-transformers' embedding models "
                    f"are supported. Got invalid 'repo_id' {repo_id}."
                )
            client = InferenceApi(
                repo_id=repo_id,
                token=huggingfacehub_api_token,
                task=values.get("task"),
            )
            if client.task not in VALID_TASKS:
                raise ValueError(
                    f"Got invalid task {client.task}, "
                    f"currently only {VALID_TASKS} are supported"
                )
            values["client"] = client
        except ImportError:
            raise ValueError(
                "Could not import huggingface_hub python package. "
                "Please install it with `pip install huggingface_hub`."
            )
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """HuggingFaceHubの埋め込みエンドポイントを呼び出して、埋め込み検索ドキュメントを取得します。

            引数:
                texts: 埋め込むテキストのリスト。

            戻り値:
                 各テキストに対する埋め込みのリスト。
        """
        # replace newlines, which can negatively affect performance.
        texts = [text.replace("\n", " ") for text in texts]
        _model_kwargs = self.model_kwargs or {}
        responses = self.client(inputs=texts, params=_model_kwargs)
        return responses

    def embed_query(self, text: str) -> List[float]:
        """HuggingFaceHubの埋め込みエンドポイントにクエリテキストの埋め込みを要求します。

                引数:
                    text: 埋め込むテキスト。

                戻り値:
                    テキストの埋め込み。
         """
        response = self.embed_documents([text])[0]
        return response
