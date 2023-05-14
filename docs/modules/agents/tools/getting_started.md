# はじめに

ツールは、エージェントが世界と対話するために使用できる機能です。
これらのツールは、汎用的なユーティリティ（例：検索）、他のチェーン、または他のエージェントであることがあります。

現在、以下のスニペットでツールを読み込むことができます。

```python
from langchain.agents import load_tools
tool_names = [...]
tools = load_tools(tool_names)
```

一部のツール（例：チェーン、エージェント）は、初期化するためにベースLLMが必要な場合があります。
その場合は、LLMも渡すことができます。

```python
from langchain.agents import load_tools
tool_names = [...]
llm = ...
tools = load_tools(tool_names, llm=llm)
```

以下は、すべてのサポートされているツールと関連情報のリストです。

- ツール名：LLMがツールを参照する名前。
- ツールの説明：LLMに渡されるツールの説明。
- ノート：LLMには渡されないツールに関するノート。
- LLMが必要か：このツールを初期化するにはLLMが必要かどうか。
- （オプション）追加のパラメータ：このツールの初期化に必要な追加のパラメータ。

## ツールのリスト

**python_repl**

- ツール名: Python REPL
- ツールの説明：Pythonのシェルです。Pythonのコマンドを実行するためにこれを使用します。入力は有効なPythonコマンドである必要があります。出力が必要な場合は、それが表示されるようにしてください。
- ノート：状態を保持します。
- LLMが必要：いいえ

**serpapi**

- ツール名: Search
- ツールの説明：検索エンジンです。現在のイベントに関する質問に答える必要があるときに便利です。入力は検索クエリである必要があります。
- ノート：Serp APIを呼び出してから結果を解析します。
- LLMが必要：いいえ

**wolfram-alpha**

- ツール名: Wolfram Alpha
- ツールの説明：Wolfram Alphaの検索エンジンです。数学、科学、技術、文化、社会、日常生活に関する質問に答える必要がある場合に便利です。入力は検索クエリである必要があります。
- ノート：Wolfram Alpha APIを呼び出してから結果を解析します。
- LLMが必要：いいえ
- 追加のパラメータ： `wolfram_alpha_appid`: Wolfram AlphaアプリのID。

**requests**

- ツール名: Requests
- ツールの説明：インターネットへのポータルです。特定のコンテンツをサイトから取得する必要があるときにこれを使用します。入力は特定のURLである必要があり、出力はそのページのすべてのテキストになります。
- ノート：Pythonのrequestsモジュールを使用します。
- LLMが必要：いいえ

**terminal**

- ツール名: Terminal
- ツールの説明：ターミナルでコマンドを実行します。入力は有効なコマンドであり、出力はそのコマンドを実行することで得られる出力です。
- ノート：subprocessでコマンドを実行します。
- LLMが必要：いいえ

**pal-math**

- ツール名: PAL-MATH
- ツールの説明：複雑なワード数学問題を解決するのに優れた言語モデル。入力は、完全に記述された難しいワード算数問題である必要があります。
- ノート：[この論文](https://arxiv.org/pdf/2211.10435.pdf)を基にしています。
- LLMが必要：はい

**pal-colored-objects**

- ツール名: PAL-COLOR-OBJ
- ツールの説明：オブジェクトの位置と色属性についての推論が素晴らしい言語モデル。入力は、完全に記述された難しい推論問題である必要があります。オブジェクトのすべての情報と、答えたい最終問題を必ず含めてください。
- ノート：[この論文](https://arxiv.org/pdf/2211.10435.pdf)を基にしています。
- LLMが必要：はい

**llm-math**

- ツール名: Calculator
- ツールの説明：数学に関する質問に答える必要がある場合に便利です。
- ノート： `LLMMath`チェーンのインスタンス。
- LLMが必要：はい

**open-meteo-api**

- ツール名: Open Meteo API
- ツールの説明: OpenMeteo APIから天候情報を取得する際に便利です。入力は、このAPIが答えられる自然言語の質問である必要があります。
- 備考: Open Meteo API (`https://api.open-meteo.com/`) の自然言語接続、特に `/v1/forecast` エンドポイント。
- LLMが必要: はい

**news-api**

- ツール名: News API
- ツールの説明: 現在のニュースのトップ見出しに関する情報を取得したい場合に使用します。入力は、このAPIが答えられる自然言語の質問である必要があります。
- 備考: News API (`https://newsapi.org`) の自然言語接続、特に `/v2/top-headlines` エンドポイント。
- LLMが必要: はい
- 追加パラメータ: `news_api_key` (このエンドポイントにアクセスするためのAPIキー)

**tmdb-api**

- ツール名: TMDB API
- ツールの説明: The Movie Databaseから情報を取得したい場合に便利です。入力は、このAPIが答えられる自然言語の質問である必要があります。
- 備考: TMDB API (`https://api.themoviedb.org/3`) の自然言語接続、特に `/search/movie` エンドポイント。
- LLMが必要: はい
- 追加パラメータ: `tmdb_bearer_token` (このエンドポイントにアクセスするためのBearerトークン - APIキーとは異なります)

**google-search**

- ツール名: 検索
- ツールの説明: Google検索のラッパーです。現在のイベントに関する質問に答える必要がある場合に便利です。入力は検索クエリであるべきです。
- 備考: Google Custom Search APIを使用
- LLMが必要: いいえ
- 追加パラメータ: `google_api_key`, `google_cse_id`
- このツールの詳細については、[このページ](../../../ecosystem/google_search.md)を参照してください

**searx-search**

- ツール名: 検索
- ツールの説明: SearxNGメタ検索エンジンのラッパーです。入力は検索クエリであるべきです。
- 備考: SearxNGは独自にデプロイが簡単です。プライバシーに優れたGoogle検索の代替手段です。SearxNG APIを使用。
- LLMが必要: いいえ
- 追加パラメータ: `searx_host`

**google-serper**

- ツール名: 検索
- ツールの説明: 低コストのGoogle検索APIです。現在のイベントに関する質問に答える必要がある場合に便利です。入力は検索クエリであるべきです。
- 備考: [serper.dev](https://serper.dev) のGoogle検索APIを呼び出し、結果を解析します。
- LLMが必要: いいえ
- 追加パラメータ: `serper_api_key`
- このツールの詳細については、[このページ](../../../ecosystem/google_serper.md)を参照してください

**wikipedia**

- ツール名: ウィキペディア
- ツールの説明: ウィキペディアのラッパーです。人物、場所、企業、歴史的出来事、その他の主題に関する一般的な質問に答える必要がある場合に便利です。入力は検索クエリであるべきです。
- 備考: MediaWiki APIを呼び出し、結果を解析するために[wikipedia](https://pypi.org/project/wikipedia/) Pythonパッケージを使用。
- LLMが必要: いいえ
- 追加パラメータ: `top_k_results`

**podcast-api**

- ツール名: Podcast API
- ツールの説明: Listen Notes Podcast APIを使ってすべてのポッドキャストやエピソードを検索します。入力は、このAPIが答えられる自然言語の質問である必要があります。
- 備考: Listen Notes Podcast API (`https://www.PodcastAPI.com`) の自然言語接続、特に `/search/` エンドポイント。
- LLMが必要: はい
- 追加パラメータ: `listen_api_key` (このエンドポイントにアクセスするためのAPIキー)

**openweathermap-api**

- ツール名: OpenWeatherMap
- ツールの説明: OpenWeatherMap APIをラップするツール。指定された場所の現在の天気情報を取得するのに便利です。入力はロケーション文字列（例: 'London,GB'）である必要があります。
- 注意: OpenWeatherMap API（https://api.openweathermap.org）に接続し、特に`/data/2.5/weather` エンドポイントを利用します。
- LLMが必要: いいえ
- 追加パラメータ: `openweathermap_api_key`（このエンドポイントにアクセスするためのAPIキー）
