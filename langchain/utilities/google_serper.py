"""Util that calls Google Search using the Serper.dev API."""
from typing import Any, Dict, Optional

import aiohttp
import requests
from pydantic.class_validators import root_validator
from pydantic.main import BaseModel

from langchain.utils import get_from_dict_or_env


class GoogleSerperAPIWrapper(BaseModel):
    """Serper.devのGoogle検索APIをラップするクラス。

        無料のAPIキーは、https://serper.dev で取得できます。

        使い方としては、環境変数``SERPER_API_KEY``にAPIキーを設定するか、
        コンストラクタに`serper_api_key`という名前のパラメータを渡してください。

        例:
            .. code-block:: python

                from langchain import GoogleSerperAPIWrapper
                google_serper = GoogleSerperAPIWrapper()
    """

    k: int = 10
    gl: str = "jp"
    hl: str = "ja"
    type: str = "search"  # search, images, places, news
    tbs: Optional[str] = None
    serper_api_key: Optional[str] = None
    aiosession: Optional[aiohttp.ClientSession] = None
    google_domain: str = "google.co.jp"

    class Config:
        """このPydanticオブジェクトの設定。"""

        arbitrary_types_allowed = True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """環境にAPIキーが存在することを確認する。"""
        serper_api_key = get_from_dict_or_env(
            values, "serper_api_key", "SERPER_API_KEY"
        )
        values["serper_api_key"] = serper_api_key

        return values

    def results(self, query: str, **kwargs: Any) -> Dict:
        """GoogleSearchでクエリを実行します。"""
        return self._google_serper_search_results(
            query,
            gl=self.gl,
            hl=self.hl,
            num=self.k,
            tbs=self.tbs,
            search_type=self.type,
            **kwargs,
        )

    def run(self, query: str, **kwargs: Any) -> str:
        """GoogleSearchでクエリを実行し、結果をパースする。"""
        results = self._google_serper_search_results(
            query,
            gl=self.gl,
            hl=self.hl,
            num=self.k,
            tbs=self.tbs,
            search_type=self.type,
            **kwargs,
        )

        return self._parse_results(results)

    async def aresults(self, query: str, **kwargs: Any) -> Dict:
        """GoogleSearchでクエリを実行します。"""
        results = await self._async_google_serper_search_results(
            query,
            gl=self.gl,
            hl=self.hl,
            num=self.k,
            search_type=self.type,
            tbs=self.tbs,
            **kwargs,
        )
        return results

    async def arun(self, query: str, **kwargs: Any) -> str:
        """GoogleSearchでクエリを実行し、結果を非同期でパースする。"""
        results = await self._async_google_serper_search_results(
            query,
            gl=self.gl,
            hl=self.hl,
            num=self.k,
            search_type=self.type,
            tbs=self.tbs,
            **kwargs,
        )

        return self._parse_results(results)

    def _parse_results(self, results: dict) -> str:
        snippets = []

        if results.get("answerBox"):
            answer_box = results.get("answerBox", {})
            if answer_box.get("answer"):
                return answer_box.get("answer")
            elif answer_box.get("snippet"):
                return answer_box.get("snippet").replace("\n", " ")
            elif answer_box.get("snippetHighlighted"):
                return ", ".join(answer_box.get("snippetHighlighted"))

        if results.get("knowledgeGraph"):
            kg = results.get("knowledgeGraph", {})
            title = kg.get("title")
            entity_type = kg.get("type")
            if entity_type:
                snippets.append(f"{title}: {entity_type}.")
            description = kg.get("description")
            if description:
                snippets.append(description)
            for attribute, value in kg.get("attributes", {}).items():
                snippets.append(f"{title} {attribute}: {value}.")

        for result in results["organic"][: self.k]:
            if "snippet" in result:
                snippets.append(result["snippet"])
            for attribute, value in result.get("attributes", {}).items():
                snippets.append(f"{attribute}: {value}.")

        if len(snippets) == 0:
            return "適切なGoogle検索結果が見つかりませんでした"

        return " ".join(snippets)

    def _google_serper_search_results(
        self, search_term: str, search_type: str = "search", **kwargs: Any
    ) -> dict:
        headers = {
            "X-API-KEY": self.serper_api_key or "",
            "Content-Type": "application/json",
        }
        params = {
            "q": search_term,
            **{key: value for key, value in kwargs.items() if value is not None},
        }
        response = requests.post(
            f"https://google.serper.dev/{search_type}", headers=headers, params=params
        )
        response.raise_for_status()
        search_results = response.json()
        return search_results

    async def _async_google_serper_search_results(
        self, search_term: str, search_type: str = "search", **kwargs: Any
    ) -> dict:
        headers = {
            "X-API-KEY": self.serper_api_key or "",
            "Content-Type": "application/json",
        }
        url = f"https://google.serper.dev/{search_type}"
        params = {
            "q": search_term,
            **{key: value for key, value in kwargs.items() if value is not None},
        }

        if not self.aiosession:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, params=params, headers=headers, raise_for_status=False
                ) as response:
                    search_results = await response.json()
        else:
            async with self.aiosession.post(
                url, params=params, headers=headers, raise_for_status=True
            ) as response:
                search_results = await response.json()

        return search_results
