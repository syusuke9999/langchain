"""SerpAPIを呼び出すチェーン。

大部分はhttps://github.com/ofirpress/self-askから借用されています。
"""
import os
import sys
from typing import Any, Dict, Optional, Tuple

import aiohttp
from pydantic import BaseModel, Extra, Field, root_validator

from langchain.utils import get_from_dict_or_env


class HiddenPrints:
    """コンテキストマネージャーでプリントを非表示にする。"""

    def __enter__(self) -> None:
        """Open file to pipe stdout to."""
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, *_: Any) -> None:
        """Close file that stdout was piped to."""
        sys.stdout.close()
        sys.stdout = self._original_stdout


class SerpAPIWrapper(BaseModel):
    """SerpAPIをラップするクラス。

        使い方：``google-search-results`` Pythonパッケージをインストールし、
        環境変数``SERPAPI_API_KEY``にAPIキーを設定するか、
        コンストラクタに`serpapi_api_key`という名前のパラメータを渡してください。

        例：
            .. code-block:: python

                from langchain import SerpAPIWrapper
                serpapi = SerpAPIWrapper()
    """

    search_engine: Any  #: :meta private:
    params: dict = Field(
        default={
            "engine": "google",
            "google_domain": "google.com",
            "gl": "jp",
            "hl": "ja",
        }
    )
    serpapi_api_key: Optional[str] = None
    aiosession: Optional[aiohttp.ClientSession] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """環境にAPIキーとPythonパッケージが存在することを確かめる。"""
        serpapi_api_key = get_from_dict_or_env(
            values, "serpapi_api_key", "SERPAPI_API_KEY"
        )
        values["serpapi_api_key"] = serpapi_api_key
        try:
            from serpapi import GoogleSearch

            values["search_engine"] = GoogleSearch
        except ImportError:
            raise ValueError(
                "Could not import serpapi python package. "
                "Please install it with `pip install google-search-results`."
            )
        return values

    async def arun(self, query: str, **kwargs: Any) -> str:
        """SerpAPIでクエリを実行し、結果を非同期でパースする。"""
        return self._process_response(await self.aresults(query))

    def run(self, query: str, **kwargs: Any) -> str:
        """SerpAPIでクエリを実行し、結果を解析します。"""
        return self._process_response(self.results(query))

    def results(self, query: str) -> dict:
        """SerpAPIでクエリを実行し、生の結果を返します"""
        params = self.get_params(query)
        with HiddenPrints():
            search = self.search_engine(params)
            res = search.get_dict()
        return res

    async def aresults(self, query: str) -> dict:
        """aiohttpを利用してSerpAPIでクエリを実行し、結果を非同期で返します。"""

        def construct_url_and_params() -> Tuple[str, Dict[str, str]]:
            params = self.get_params(query)
            params["source"] = "python"
            if self.serpapi_api_key:
                params["serp_api_key"] = self.serpapi_api_key
            params["output"] = "json"
            url = "https://serpapi.com/search"
            return url, params

        url, params = construct_url_and_params()
        if not self.aiosession:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    res = await response.json()
        else:
            async with self.aiosession.get(url, params=params) as response:
                res = await response.json()

        return res

    def get_params(self, query: str) -> Dict[str, str]:
        """SerpAPI用のパラメータを取得します。"""
        _params = {
            "api_key": self.serpapi_api_key,
            "q": query,
        }
        params = {**self.params, **_params}
        return params

    @staticmethod
    def _process_response(res: dict) -> str:
        """SerpAPIからの応答を処理します。"""
        if "error" in res.keys():
            raise ValueError(f"Got error from SerpAPI: {res['error']}")
        if "answer_box" in res.keys() and "answer" in res["answer_box"].keys():
            toret = res["answer_box"]["answer"]
        elif "answer_box" in res.keys() and "snippet" in res["answer_box"].keys():
            toret = res["answer_box"]["snippet"]
        elif (
            "answer_box" in res.keys()
            and "snippet_highlighted_words" in res["answer_box"].keys()
        ):
            toret = res["answer_box"]["snippet_highlighted_words"][0]
        elif (
            "sports_results" in res.keys()
            and "game_spotlight" in res["sports_results"].keys()
        ):
            toret = res["sports_results"]["game_spotlight"]
        elif (
            "knowledge_graph" in res.keys()
            and "description" in res["knowledge_graph"].keys()
        ):
            toret = res["knowledge_graph"]["description"]
        elif "snippet" in res["organic_results"][0].keys():
            toret = res["organic_results"][0]["snippet"]

        else:
            toret = "No good search result found"
        return toret
