"""OpenAIの埋め込みモデルをラップしたクラスです"""
from __future__ import annotations

import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
from pydantic import BaseModel, Extra, root_validator
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


def _create_retry_decorator(embeddings: OpenAIEmbeddings) -> Callable[[Any], Any]:
    import openai

    min_seconds = 4
    max_seconds = 10
    # 各リトライの間に2^x * 1秒待機します。
    # 最初は4秒、次に最大10秒、その後は10秒間隔で待機します。
    return retry(
        reraise=True,
        stop=stop_after_attempt(embeddings.max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
                retry_if_exception_type(openai.error.Timeout)
                | retry_if_exception_type(openai.error.APIError)
                | retry_if_exception_type(openai.error.APIConnectionError)
                | retry_if_exception_type(openai.error.RateLimitError)
                | retry_if_exception_type(openai.error.ServiceUnavailableError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def _async_retry_decorator(embeddings: OpenAIEmbeddings) -> Any:
    import openai

    min_seconds = 4
    max_seconds = 10
    # 各リトライの間に2^x * 1秒待機します。
    # 最初は4秒、次に最大10秒、その後は10秒間隔で待機します。
    async_retrying = AsyncRetrying(
        reraise=True,
        stop=stop_after_attempt(embeddings.max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
                retry_if_exception_type(openai.error.Timeout)
                | retry_if_exception_type(openai.error.APIError)
                | retry_if_exception_type(openai.error.APIConnectionError)
                | retry_if_exception_type(openai.error.RateLimitError)
                | retry_if_exception_type(openai.error.ServiceUnavailableError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )

    def wrap(func: Callable) -> Callable:
        async def wrapped_f(*args: Any, **kwargs: Any) -> Callable:
            async for _ in async_retrying:
                return await func(*args, **kwargs)
            raise AssertionError("this is unreachable")

        return wrapped_f

    return wrap


def embed_with_retry(embeddings: OpenAIEmbeddings, **kwargs: Any) -> Any:
    """embeddingの呼び出しを再試行するためにtenacityを使用します"""
    retry_decorator = _create_retry_decorator(embeddings)

    @retry_decorator
    def _embed_with_retry(**kwargs: Any) -> Any:
        return embeddings.client.create(**kwargs)

    return _embed_with_retry(**kwargs)


async def async_embed_with_retry(embeddings: OpenAIEmbeddings, **kwargs: Any) -> Any:
    """embeddingの呼び出しを再試行するためにtenacityを使用します"""

    @_async_retry_decorator(embeddings)
    async def _async_embed_with_retry(**kwargs: Any) -> Any:
        return await embeddings.client.acreate(**kwargs)

    return await _async_embed_with_retry(**kwargs)


class OpenAIEmbeddings(BaseModel, Embeddings):
    """OpenAI埋め込みモデルのラッパークラスです。

        使用するには、``openai``パッケージがインストールされている必要があります。
        環境変数``OPENAI_API_KEY``にAPIキーを設定するか、コンストラクタに名前付きパラメータとして渡してください。

        例:
            .. code-block:: python

                from langchain.embeddings import OpenAIEmbeddings
                openai = OpenAIEmbeddings(openai_api_key="my-api-key")

        Microsoft Azureエンドポイントでライブラリを使用するためには、
        OPENAI_API_TYPE、OPENAI_API_BASE、OPENAI_API_KEY、OPENAI_API_VERSIONを設定する必要があります。
        OPENAI_API_TYPEは'azure'に設定し、他のパラメータはエンドポイントのプロパティに対応します。
        さらに、モデルパラメータとしてデプロイメント名を渡す必要があります。

        例:
            .. code-block:: python

                import os
                os.environ["OPENAI_API_TYPE"] = "azure"
                os.environ["OPENAI_API_BASE"] = "https://<your-endpoint.openai.azure.com/"
                os.environ["OPENAI_API_KEY"] = "your AzureOpenAI key"
                os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
                os.environ["OPENAI_PROXY"] = "http://your-corporate-proxy:8080"

                from langchain.embeddings.openai import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings(
                    deployment="your-embeddings-deployment-name",
                    model="your-embeddings-model-name",
                    openai_api_base="https://your-endpoint.openai.azure.com/",
                    openai_api_type="azure",
                )
                text = "This is a test query."
                query_result = embeddings.embed_query(text)

    """

    client: Any  #: :meta private:
    model: str = "text-embedding-ada-002"
    deployment: str = model  # to support Azure OpenAI Service custom deployment names
    openai_api_version: Optional[str] = None
    # to support Azure OpenAI Service custom endpoints
    openai_api_base: Optional[str] = None
    # to support Azure OpenAI Service custom endpoints
    openai_api_type: Optional[str] = None
    # to support explicit proxy for OpenAI
    openai_proxy: Optional[str] = None
    embedding_ctx_length: int = 8191
    openai_api_key: Optional[str] = None
    openai_organization: Optional[str] = None
    allowed_special: Union[Literal["all"], Set[str]] = set()
    disallowed_special: Union[Literal["all"], Set[str], Sequence[str]] = "all"
    chunk_size: int = 1500
    """各バッチで埋め込むテキストの最大数"""
    max_retries: int = 20
    """生成時に行う最大リトライ回数"""
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    """OpenAPIリクエストのタイムアウト（秒）"""
    headers: Any = None
    tiktoken_model_name: Optional[str] = None
    """このクラスを使用する際にtiktokenに渡すモデル名。
        Tiktokenは、ドキュメントのトークン数を数えて特定の制限以下に抑えるために使用されます。
        デフォルトでNoneに設定されている場合、これは埋め込みモデルの名前と同じになります。
        しかし、tiktokenがサポートしていないモデル名でこのEmbeddingクラスを使用したい場合があります。
        これには、Azureの埋め込みを使用する場合や、OpenAIのようなAPIを公開しているがモデルが異なる多くのモデルプロバイダを使用する場合が含まれます。
        これらのケースでは、tiktokenが呼び出されたときにエラーを避けるために、ここで使用するモデル名を指定できます。"""

    class Config:
        """このpydanticオブジェクトの設定"""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """環境内にAPIキーとPythonパッケージが存在することを確認します。"""
        values["openai_api_key"] = get_from_dict_or_env(
            values, "openai_api_key", "OPENAI_API_KEY"
        )
        values["openai_api_base"] = get_from_dict_or_env(
            values,
            "openai_api_base",
            "OPENAI_API_BASE",
            default="",
        )
        values["openai_api_type"] = get_from_dict_or_env(
            values,
            "openai_api_type",
            "OPENAI_API_TYPE",
            default="",
        )
        values["openai_proxy"] = get_from_dict_or_env(
            values,
            "openai_proxy",
            "OPENAI_PROXY",
            default="",
        )
        if values["openai_api_type"] in ("azure", "azure_ad", "azuread"):
            default_api_version = "2022-12-01"
        else:
            default_api_version = ""
        values["openai_api_version"] = get_from_dict_or_env(
            values,
            "openai_api_version",
            "OPENAI_API_VERSION",
            default=default_api_version,
        )
        values["openai_organization"] = get_from_dict_or_env(
            values,
            "openai_organization",
            "OPENAI_ORGANIZATION",
            default="",
        )
        try:
            import openai

            values["client"] = openai.Embedding
        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        return values

    @property
    def _invocation_params(self) -> Dict:
        openai_args = {
            "engine": self.deployment,
            "request_timeout": self.request_timeout,
            "headers": self.headers,
            "api_key": self.openai_api_key,
            "organization": self.openai_organization,
            "api_base": self.openai_api_base,
            "api_type": self.openai_api_type,
            "api_version": self.openai_api_version,
        }
        if self.openai_proxy:
            import openai

            openai.proxy = {
                "http": self.openai_proxy,
                "https": self.openai_proxy,
            }  # type: ignore[assignment]  # noqa: E501
        return openai_args

    # 詳細は以下を参照してください
    # https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
    def _get_len_safe_embeddings(
            self, texts: List[str], *, engine: str, chunk_size: Optional[int] = None
    ) -> List[List[float]]:
        embeddings: List[List[float]] = [[] for _ in range(len(texts))]
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "Could not import tiktoken python package. "
                "This is needed in order to for OpenAIEmbeddings. "
                "Please install it with `pip install tiktoken`."
            )

        tokens = []
        indices = []
        model_name = self.tiktoken_model_name or self.model
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            logger.warning("Warning: model not found. Using cl100k_base encoding.")
            model = "cl100k_base"
            encoding = tiktoken.get_encoding(model)
        for i, text in enumerate(texts):
            if self.model.endswith("001"):
                # 参照: https://github.com/openai/openai-python/issues/418#issuecomment-1525939500
                # パフォーマンスに悪影響を与える可能性のある改行を置換します。
                text = text.replace("\n", " ")
            token = encoding.encode(
                text,
                allowed_special=self.allowed_special,
                disallowed_special=self.disallowed_special,
            )
            for j in range(0, len(token), self.embedding_ctx_length):
                tokens += [token[j: j + self.embedding_ctx_length]]
                indices += [i]

        batched_embeddings = []
        _chunk_size = chunk_size or self.chunk_size
        for i in range(0, len(tokens), _chunk_size):
            response = embed_with_retry(
                self,
                input=tokens[i: i + _chunk_size],
                **self._invocation_params,
            )
            batched_embeddings += [r["embedding"] for r in response["data"]]

        results: List[List[List[float]]] = [[] for _ in range(len(texts))]
        num_tokens_in_batch: List[List[int]] = [[] for _ in range(len(texts))]
        for i in range(len(indices)):
            results[indices[i]].append(batched_embeddings[i])
            num_tokens_in_batch[indices[i]].append(len(tokens[i]))

        for i in range(len(texts)):
            _result = results[i]
            if len(_result) == 0:
                average = embed_with_retry(
                    self,
                    input="",
                    **self._invocation_params,
                )[
                    "data"
                ][0]["embedding"]
            else:
                average = np.average(_result, axis=0, weights=num_tokens_in_batch[i])
            embeddings[i] = (average / np.linalg.norm(average)).tolist()

        return embeddings

    # 以下を参照してください
    # https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
    async def _aget_len_safe_embeddings(
            self, texts: List[str], *, engine: str, chunk_size: Optional[int] = None
    ) -> List[List[float]]:
        embeddings: List[List[float]] = [[] for _ in range(len(texts))]
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "Could not import tiktoken python package. "
                "This is needed in order to for OpenAIEmbeddings. "
                "Please install it with `pip install tiktoken`."
            )

        tokens = []
        indices = []
        model_name = self.tiktoken_model_name or self.model
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            logger.warning("Warning: model not found. Using cl100k_base encoding.")
            model = "cl100k_base"
            encoding = tiktoken.get_encoding(model)
        for i, text in enumerate(texts):
            if self.model.endswith("001"):
                # 参照: https://github.com/openai/openai-python/issues/418#issuecomment-1525939500
                # パフォーマンスに悪影響を与える可能性のある改行を置換します。
                text = text.replace("\n", " ")
            token = encoding.encode(
                text,
                allowed_special=self.allowed_special,
                disallowed_special=self.disallowed_special,
            )
            for j in range(0, len(token), self.embedding_ctx_length):
                tokens += [token[j: j + self.embedding_ctx_length]]
                indices += [i]

        batched_embeddings = []
        _chunk_size = chunk_size or self.chunk_size
        for i in range(0, len(tokens), _chunk_size):
            response = await async_embed_with_retry(
                self,
                input=tokens[i: i + _chunk_size],
                **self._invocation_params,
            )
            batched_embeddings += [r["embedding"] for r in response["data"]]

        results: List[List[List[float]]] = [[] for _ in range(len(texts))]
        num_tokens_in_batch: List[List[int]] = [[] for _ in range(len(texts))]
        for i in range(len(indices)):
            results[indices[i]].append(batched_embeddings[i])
            num_tokens_in_batch[indices[i]].append(len(tokens[i]))

        for i in range(len(texts)):
            _result = results[i]
            if len(_result) == 0:
                average = (
                    await async_embed_with_retry(
                        self,
                        input="",
                        **self._invocation_params,
                    )
                )["data"][0]["embedding"]
            else:
                average = np.average(_result, axis=0, weights=num_tokens_in_batch[i])
            embeddings[i] = (average / np.linalg.norm(average)).tolist()

        return embeddings

    def _embedding_func(self, text: str, *, engine: str) -> List[float]:
        """OpenAIの埋め込みエンドポイントに呼び出しを行います。"""
        # handle large input text
        if len(text) > self.embedding_ctx_length:
            return self._get_len_safe_embeddings([text], engine=engine)[0]
        else:
            if self.model.endswith("001"):
                # 参照: https://github.com/openai/openai-python/issues/418#issuecomment-1525939500
                # パフォーマンスに悪影響を与える可能性のある改行を置換します。
                text = text.replace("\n", " ")
            return embed_with_retry(
                self,
                input=[text],
                **self._invocation_params,
            )[
                "data"
            ][0]["embedding"]

    async def _aembedding_func(self, text: str, *, engine: str) -> List[float]:
        """OpenAIの埋め込みエンドポイントに呼び出しを行います。"""
        # 大きな入力テキストを処理する
        if len(text) > self.embedding_ctx_length:
            return (await self._aget_len_safe_embeddings([text], engine=engine))[0]
        else:
            if self.model.endswith("001"):
                # 参照: https://github.com/openai/openai-python/issues/418#issuecomment-1525939500
                # パフォーマンスに悪影響を与える可能性のある改行を置換します。
                text = text.replace("\n", " ")
            return (
                await async_embed_with_retry(
                    self,
                    input=[text],
                    **self._invocation_params,
                )
            )["data"][0]["embedding"]

    def embed_documents(
            self, texts: List[str], chunk_size: Optional[int] = 0
    ) -> List[List[float]]:
        """埋め込み検索ドキュメントのためにOpenAIの埋め込みエンドポイントに呼び出しを行います。

                Args:
                    texts: 埋め込むテキストのリスト。
                    chunk_size: 埋め込みのチャンクサイズ。Noneの場合、クラスで指定されたチャンクサイズが使用されます。

                Returns:
                    テキストごとの埋め込みのリスト。
        """
        # 注意: 簡単にするため、リストには最大コンテキストよりも長いテキストが含まれる可能性があると仮定し、
        # 長さに安全な埋め込み関数を使用します。
        return self._get_len_safe_embeddings(texts, engine=self.deployment)

    async def aembed_documents(
            self, texts: List[str], chunk_size: Optional[int] = 0
    ) -> List[List[float]]:
        """埋め込み検索ドキュメントのためにOpenAIの非同期埋め込みエンドポイントに呼び出しを行います。

                Args:
                    texts: 埋め込むテキストのリスト。
                    chunk_size: 埋め込みのチャンクサイズ。Noneの場合、クラスで指定されたチャンクサイズが使用されます。

                Returns:
                    テキストごとの埋め込みのリスト。
        """
        # 注意: 簡単にするため、リストには最大コンテキストよりも長いテキストが含まれる可能性があると仮定し、
        # 長さに安全な埋め込み関数を使用します。
        return await self._aget_len_safe_embeddings(texts, engine=self.deployment)

    def embed_query(self, text: str) -> List[float]:
        """埋め込みクエリテキストのためにOpenAIの埋め込みエンドポイントに呼び出しを行います。

                Args:
                    text: 埋め込むテキスト。

                Returns:
                    テキストの埋め込み。
        """
        embedding = self._embedding_func(text, engine=self.deployment)
        return embedding

    async def aembed_query(self, text: str) -> List[float]:
        """埋め込みクエリテキストのためにOpenAIの非同期埋め込みエンドポイントに呼び出しを行います。

                Args:
                    text: 埋め込むテキスト。

                Returns:
                    テキストの埋め込み。
        """
        embedding = await self._aembedding_func(text, engine=self.deployment)
        return embedding
