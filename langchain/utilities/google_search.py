"""Google検索を呼び出すユーティリティ"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Extra, root_validator

from langchain.utils import get_from_dict_or_env


class GoogleSearchAPIWrapper(BaseModel):
    """Google Search APIのラッパー。

        以下の指示に従って適応させました：https://stackoverflow.com/questions/
        37083058/
        programmatically-searching-google-in-python-using-custom-search

        TODO: 使い方のドキュメント
        1. google-api-python-clientをインストール
        - まだGoogleアカウントを持っていない場合は、登録してください。
        - Google API Consoleでプロジェクトを作成したことがない場合は、
        プロジェクト管理ページを読んで、Google API Consoleでプロジェクトを作成します。
        - ライブラリをpip install google-api-python-clientでインストールします。
        現在のライブラリのバージョンは、この時点で2.70.0です。

        2. APIキーの作成方法：
        - Cloud ConsoleのAPI＆サービス→認証情報パネルに移動します。
        - 認証情報を作成を選択し、ドロップダウンメニューからAPIキーを選択します。
        - APIキーが作成されたダイアログボックスに、新しく作成されたキーが表示されます。
        - これでAPI_KEYを取得しました。

        3. カスタム検索エンジンを設定して、Web全体を検索できるようにする
        - このリンクでカスタム検索エンジンを作成します。
        - 検索対象のサイトに、有効なURLを追加します（例：www.stackoverflow.com）。
        - それ以外は入力する必要はありません。
        左側のメニューで、検索エンジンの編集→{検索エンジン名}→設定をクリックし、
        Web全体を検索をONに設定します。検索対象のサイトから追加したURLを削除します。
        - 検索エンジンIDの下に、検索エンジンIDが表示されます。

        4. カスタム検索APIを有効にする
        - Cloud ConsoleのAPI＆サービス→ダッシュボードパネルに移動します。
        - APIとサービスを有効にするをクリックします。
        - カスタム検索APIを検索し、クリックします。
        - 有効にするをクリックします。
        それのURL：https://console.cloud.google.com/apis/library/customsearch.googleapis.com
    """

    search_engine: Any  #: :meta private:
    google_api_key: Optional[str] = None
    google_cse_id: Optional[str] = None
    k: int = 10
    siterestrict: bool = False

    class Config:
        """当該pydanticオブジェクトの設定。"""
        extra = Extra.forbid

    def _google_search_results(self, search_term: str, **kwargs: Any) -> List[dict]:
        cse = self.search_engine.cse()
        if self.siterestrict:
            cse = cse.siterestrict()
        res = cse.list(q=search_term, cx=self.google_cse_id, **kwargs).execute()
        return res.get("items", [])

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """環境にAPIキーとPythonパッケージが存在することを確認する。"""
        google_api_key = get_from_dict_or_env(
            values, "google_api_key", "GOOGLE_API_KEY"
        )
        values["google_api_key"] = google_api_key

        google_cse_id = get_from_dict_or_env(values, "google_cse_id", "GOOGLE_CSE_ID")
        values["google_cse_id"] = google_cse_id

        try:
            from googleapiclient.discovery import build

        except ImportError:
            raise ImportError(
                "google-api-python-client is not installed. "
                "Please install it with `pip install google-api-python-client`"
            )

        service = build("customsearch", "v1", developerKey=google_api_key)
        values["search_engine"] = service

        return values

    def run(self, query: str) -> str:
        """GoogleSearchでクエリを実行し、結果を解析する。"""
        snippets = []
        results = self._google_search_results(query, num=self.k)
        if len(results) == 0:
            return "No good Google Search Result was found"
        for result in results:
            if "snippet" in result:
                snippets.append(result["snippet"])

        return " ".join(snippets)

    def results(self, query: str, num_results: int) -> List[Dict]:
        """GoogleSearchを通じてクエリを実行し、メタデータを返します。
                引数:
                    query: 検索するクエリ。
                    num_results: 返す結果の数。
                戻り値:
                    以下のキーを持つ辞書のリスト:
                        snippet - 結果の説明。
                        title - 結果のタイトル。
                        link - 結果へのリンク。
        """
        metadata_results = []
        results = self._google_search_results(query, num=num_results)
        if len(results) == 0:
            return [{"Result": "No good Google Search Result was found"}]
        for result in results:
            metadata_result = {
                "title": result["title"],
                "link": result["link"],
            }
            if "snippet" in result:
                metadata_result["snippet"] = result["snippet"]
            metadata_results.append(metadata_result)

        return metadata_results
