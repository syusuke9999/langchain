# クイックスタートガイド

このチュートリアルでは、LangChainを使ったエンドツーエンドの言語モデルアプリケーションの構築方法について簡単に説明します。

## インストール

まず始めに、以下のコマンドでLangChainをインストールしてください：

```bash
pip install langchain
# または
conda install langchain -c conda-forge
```

## 環境設定

LangChainを使用するには通常、1つ以上のモデルプロバイダ、データストア、APIなどとの統合が必要です。

このサンプルでは、OpenAIのAPIを使用するので、まず同社のSDKをインストールする必要があります：

```bash
pip install openai
```

次に、ターミナルで環境変数を設定する必要があります。

```bash
export OPENAI_API_KEY="..."
```

あるいは、Jupyterノートブック（またはPythonスクリプト）の内部から行うこともできます：

```python
import os
os.environ["OPENAI_API_KEY"] = "..."
```

## 言語モデルアプリケーションの構築： LLMsの場合

LangChainをインストールし、環境を整えたところで、言語モデルアプリケーションの構築を開始してみましょう。

LangChainは、言語モデルのアプリケーションを構築するために利用できる、多くのモジュールを提供しています。 モジュールを組み合わせてより複雑なアプリケーションを作成することもできますし、個別に使用してシンプルなアプリケーションを作成することもできます。

## LLMs: 言語モデルから予測を得る

LangChainの最も基本的な構成要素は、ある入力に対してLLMを呼び出すことです。
では、その方法を簡単なサンプルで説明しましょう。
ここでは、会社が作っているものに応じた会社名を生成するサービスを作ることにしましょう。

そのためには、まずLLMラッパーをインポートする必要があります。

```python
from langchain.llms import OpenAI
```

そして、任意の引数でラッパーを初期化することができます。
このサンプルでは、おそらく出力がよりランダムになるようにしたいので、高い温度で初期化することにします。

```python
llm = OpenAI(temperature=0.9)
```

これで、入力に対して呼び出すことができます！

```python
text = "カラフルな靴下を作る会社の社名として、何かいいものはありますか？"
print(llm(text))
```

```pycon
楽しい足元
```

LangChain内でLLMを使用する方法の詳細については、[LLMのGetting Startedガイド](../modules/models/llms/getting_started.ipynb)を参照してください。


## プロンプトテンプレート: LLM用のプロンプトを管理する

LLMを呼び出すことは素晴らしい第一歩ですが、それだけでは始まりに過ぎません。
通常LLMをアプリケーションで使用する場合、ユーザー入力を直接LLMに送信するのではなく、
ユーザー入力を受け取ってプロンプトを構築し、それをLLMに送信することが多いでしょう。

例えば前の例では、渡したテキストはカラフルな靴下を作る会社の名前を尋ねるようにハードコーディングされていました。
この仮想的なサービスでは会社が何をするのかを説明するユーザー入力だけを受け取り、その情報を使ってプロンプトをフォーマットすることが望ましいでしょう。

これはLangChainを使えば簡単にできます！

まずプロンプトテンプレートを定義しましょう：

```python
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["product"],
    template="{product}を作る会社の社名として、何かいいものはありますか？",
)
```

これがどのように機能するか見てみましょう！`.format`メソッドを使ってフォーマットできます。

```python
print(prompt.format(product="カラフルな靴下"))
```

```pycon
カラフルな靴下を作るを作る会社の社名として、何かいいものはありますか？
```


[詳細については、プロンプトの入門ガイドをご覧ください。](../modules/prompts/chat_prompt_template.ipynb)

## チェーン: LLMとプロンプトを複数ステップのワークフローで組み合わせる

これまでPromptTemplateとLLMというプリミティブを個別に扱ってきました。しかし実際のアプリケーションは単一のプリミティブだけでなく、それらの組み合わせで構成されています。

LangChainのチェーンは、LLMのようなプリミティブや他のチェーンからなるリンクで構成されています。

最も基本的なタイプのチェーンはLLMChainで、PromptTemplateとLLMから構成されています。

前の例を拡張してユーザー入力を受け取りPromptTemplateでフォーマットし、フォーマットされた応答をLLMに渡すLLMChainを構築できます。

```python
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["product"],
    template="{product}を作る会社の社名として、何かいいものはありますか？",
)
```

これで、ユーザーの入力を受け取り、それを使ってプロンプトを形式化し、LLMに送信するという、非常にシンプルなチェーンを作れるようになりました:

```python
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)
```

これで製品を指定するだけで、このチェーンを走らせることができるようになりました！

```python
chain.run("カラフルな靴下")
# -> '\n\nSocktastic!'
```

では行きましょう！まず1つ目のチェーン、LLMチェーンがあります。
チェーンのタイプとしては単純な部類に入りますが、その仕組みを理解することでより複雑なチェーンに対応することができます。

[詳細については、チェーンの入門ガイドをご覧ください。](../modules/chains/getting_started.ipynb)

## エージェント: ユーザー入力に基づいて動的にチェーンを呼び出す

これまで見てきたチェーンは、あらかじめ決められた順序で実行されます。

エージェントはもはやそうではありません。LLMを使用してどのアクションを実行しどの順序で実行するか決定します。アクションは、ツールを使用してその出力を観察するか、ユーザーに戻るかのいずれかです。

正しく使用するとエージェントは非常に強力になります。このチュートリアルでは最も簡単で最も高レベルなAPIを使って、エージェントを簡単に使う方法を紹介します。


エージェントをロードするには、以下の概念を理解する必要があります。

- ツール: 特定の任務を遂行する関数のこと。 これは例えば、次のようなものがあります： Google検索、データベース検索、Python REPLその他のチェーンなど。現在、ツールのインターフェースは文字列を入力とし文字列を出力として想定される関数です。
- LLM: エージェントを動かす言語モデル。
- エージェント: 使用するエージェント。これは、サポートしているエージェントクラスを指し示す文字列でなければなりません。このノートブックでは、最もシンプルで高レベルのAPIに焦点を当てているため、標準的にサポートされているエージェントの使用のみを取り上げています。カスタムエージェントを実装したい場合は、カスタムエージェントのドキュメントを参照してください（近日公開予定）。

**エージェント**: サポートされているエージェントとその仕様のリストについては[こちら](../modules/agents/getting_started.ipynb)を参照してください。

**ツール**: 事前に定義されたツールとその仕様のリストについては[こちら](../modules/agents/tools/getting_started.md)を参照してください。

この例ではSerpAPI Pythonパッケージもインストールする必要があります。

```bash
pip install google-search-results
```

そして適切な環境変数を設定します。

```python
import os
os.environ["SERPAPI_API_KEY"] = "..."
```

これで始められます！

```python
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

# まずエージェントを制御するために使用する言語モデルをロードしましょう。
llm = OpenAI(temperature=0)

# 次に使用するツールをロードしましょう。`llm-math`ツールはLLMを使用するためそれを渡す必要があります。
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# 最後に、ツール、言語モデル、使用するエージェントのタイプを使ってエージェントを初期化しましょう。
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# それでは試してみましょう！
agent.run("昨日のサンフランシスコの最高気温は華氏何度でしたか？その数値を0.023乗したらどうなりますか？")
```

```pycon
> 新しいAgentExecutorチェーンに入ります...
 まず温度を求め、それを電卓で0.023乗する必要があります。
アクション: 検索
アクション 入力: "昨日のサンフランシスコの最高気温"
観察: サンフランシスコの昨日の気温。最高気温：57°F（午後1時56分）最低気温：49°F（午前1時56分）平均気温 ...
思考: 気温が分かったので、計算機を使って0.023乗できます。
アクション: 計算機
アクション入力: 57^0.023
観察: 回答: 1.0974509573251117

思考: 最終的な答えが分かりました
最終的な解答: 昨日のサンフランシスコの最高気温（華氏）を0.023乗した値は1.0974509573251117です。

> チェーンが終了しました。
```


## メモリ： チェーンとエージェントに状態を持たせる

今までのチェーンやエージェントは、すべてステートレス（状態を持たない）でした。 しかし、チェーンやエージェントに「記憶」の概念を持たせて、以前の対話に関する情報を記憶させたいと思うことがよくあります。例えば、チャットボットを設計する場合、以前のメッセージを記憶させ、そこから得られるコンテキストを利用してより良い会話を行えるようにする、というのが最も明確でシンプルな例です。 これは「短期記憶」の一種と言えるでしょう。もっと複雑な例では、チェーンやエージェントが重要な情報を長期的に記憶していくようなイメージです。後者に関するより具体的なアイデアについては、この[素晴らしい論文](https://memprompt.com/)を参照してください。

LangChainはこの目的のために特別に作られたチェーンをいくつか提供しています。このノートブックでは、これらのチェーンの一つ（`ConversationChain`）を2つの異なるタイプのメモリで使用する方法を説明します。

デフォルトでは、`ConversationChain`は単純なタイプのメモリを持っており、以前のすべての入出力を記憶して、渡されたコンテキストに追加します。このチェーンの使い方を見てみましょう（プロンプトが表示されるように `verbose=True` をセットします）。

```python
from langchain import OpenAI, ConversationChain

llm = OpenAI(temperature=0)
conversation = ConversationChain(llm=llm, verbose=True)

output = conversation.predict(input="やあ、こんにちは！")
print(output)
```

```pycon
> Entering new chain...
Prompt after formatting:
以下は、人間とAIのフレンドリーな 会話です。AIはおしゃべりで、コンテキストから具体的な情報をたくさん提供してくれます。AIは質問の答えを知らない場合、素直に「分かりません」と答えます。

Current conversation:

Human: やあ、こんにちは！
AI:

> Finished chain.
' こんにちは！今日はお元気ですか？'
```

```python
output = conversation.predict(input="順調にやってます！ちょうどAIと会話しているところです。")
print(output)
```

```pycon
> Entering new chain...
Prompt after formatting:
以下は、人間とAIのフレンドリーな 会話です。AIはおしゃべりで、コンテキストから具体的な情報をたくさん提供してくれます。AIは質問の答えを知らない場合、素直に「分かりません」と答えます。

Current conversation:

Human: やあ、こんにちは！
AI:  こんにちは！今日はお元気ですか？
Human: 順調にやってます！ちょうどAIと会話しているところです。
AI:

> Finished chain.
" それは素晴らしいですね！何について話しましょうか？"
```

## 言語モデルアプリケーションの構築: チャットモデル

同様にLLMの代わりにチャットモデルを使用することもできます。チャットモデルは言語モデルの一種です。チャットモデルは言語モデルを内部で使用していますが、公開されているインターフェースは少し異なります。テキスト入力、テキスト出力のAPIではなく、チャットメッセージが入力と出力となるインターフェースが提供されています。

チャットモデルAPIは比較的新しいため、適切な抽象化をまだ見つけている最中です。

## チャットモデルからメッセージの補完を取得する

チャットモデルに1つ以上のメッセージを渡すことでチャットの補完を取得できます。応答はメッセージになります。LangChainで現在サポートされているメッセージのタイプは、`AIMessage`、`HumanMessage`、`SystemMessage`、`ChatMessage`です。`ChatMessage`は任意のロールパラメータを取ります。ほとんどの場合、`HumanMessage`、`AIMessage`、`SystemMessage`を扱うことになります。

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

chat = ChatOpenAI(temperature=0)
```

1つのメッセージを渡すことで補完を取得できます。

```python
chat([HumanMessage(content="この文章を英語から日本語に訳してください。私はプログラミングが大好きです。")])
# -> AIMessage(content="J'aime programmer.", additional_kwargs={})
```

OpenAIのgpt-3.5-turboとgpt-4モデルでは、複数のメッセージを渡すこともできます。

```python
messages = [
    SystemMessage(content="あなたは、英語を日本語に翻訳してくれる役に立つアシスタントです。"),
    HumanMessage(content="I love programming.")
]
chat(messages)
# -> AIMessage(content="J'aime programmer.", additional_kwargs={})
```

さらに一歩進めて`generate`を使って複数のメッセージセットに対して補完を生成することができます。これは追加の`message`パラメータを持つ`LLMResult`を返します。

```python
batch_messages = [
    [
        SystemMessage(content="あなたは、英語を日本語に翻訳する役に立つアシスタントです。"),
        HumanMessage(content="I love programming.")
    ],
    [
        SystemMessage(content="あなたは、英語を日本語に翻訳する役に立つアシスタントです。"),
        HumanMessage(content"I love artificial intelligence.")
    ],
]
result = chat.generate(batch_messages)
result
# -> LLMResult(generations=[[ChatGeneration(text="J'aime programmer.", generation_info=None, message=AIMessage(content="J'aime programmer.", additional_kwargs={}))], [ChatGeneration(text="J'aime l'intelligence artificielle.", generation_info=None, message=AIMessage(content="J'aime l'intelligence artificielle.", additional_kwargs={}))]], llm_output={'token_usage': {'prompt_tokens': 57, 'completion_tokens': 20, 'total_tokens': 77}})
```

このLLMResultからトークンの使用状況などを復元することができます：

```
result.llm_output['token_usage']
# -> {'prompt_tokens': 57, 'completion_tokens': 20, 'total_tokens': 77}
```

## チャットプロンプトテンプレート
LLMと同様に、`MessagePromptTemplate`を使用してテンプレート化を活用できます。1つ以上の`MessagePromptTemplate`から`ChatPromptTemplate`を構築できます。`ChatPromptTemplate`の`format_prompt`を使用することができます。これは`PromptValue`を返し、フォーマットされた値をllmまたはチャットモデルへの入力として使用する場合に、文字列または`Message`オブジェクトに変換できます。

便宜上テンプレートには`from_template`メソッドが公開されています。このテンプレートを使用する場合以下のようになります。

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

chat = ChatOpenAI(temperature=0)

template = "あなたは、{input_language}を{output_language}に翻訳する役に立つアシスタントです。"
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# get a chat completion from the formatted messages
chat(chat_prompt.format_prompt(input_language="English", output_language="Japanese", text="I love programming.").to_messages())
# -> AIMessage(content="J'aime programmer.", additional_kwargs={})
```

## チャットモデルとの連鎖
上記のセクションで説明した `LLMChain` は、チャットモデルとも使用することができます。

```python
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

chat = ChatOpenAI(temperature=0)

template = "あなたは、{input_language}を{output_language}に翻訳する役に立つアシスタントです。"
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

chain = LLMChain(llm=chat, prompt=chat_prompt)
chain.run(input_language="English", output_language="Japanese", text="I love programming.")
# -> "J'aime programmer."
```

## チャットモデルを搭載したエージェント
エージェントはチャットモデルと組み合わせて利用することもできます。エージェントタイプとして `AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION` を用いて初期化できます。

```python
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

# まずエージェントを制御するために使用する言語モデルをロードします。
chat = ChatOpenAI(temperature=0)

# 次に使用するツールをロードします。`llm-math`ツールはLLMを使用するため、それを渡す必要があります。
llm = OpenAI(temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)


# 最後にツール、言語モデル、および使用するエージェントのタイプを初期化します。
agent = initialize_agent(tools, chat, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# これでテストしてみましょう！
agent.run("オリビア・ワイルドの彼氏は誰ですか？ 彼の現在の年齢を0.23乗したものは何ですか？")
```

```pycon
> 新しいAgentExecutorチェーンに入る...
思考：オリビア・ワイルドのボーイフレンドを検索し、彼の年齢を0.23乗するために電卓を使う必要がある。
アクション：
{
  "action": "Search",
  "action_input": "オリビア・ワイルドのボーイフレンド"
}

観察：スタイルズとワイルドの関係は2020年11月に終わりました。 ワイルドはシネコン2022で『Don't Worry Darling』を発表している最中に、子供の親権に関する裁判所の書類を提出されたことが公になっている。 2021年1月、ワイルドは『Don't Worry Darling』の撮影中に出会った歌手のハリー・スタイルズと交際を始めた。
思考：ハリー・スタイルズの現在の年齢を検索する必要がある。
アクション：
{
  "action": "Search",
  "action_input": "Harry Styles age"
}

観察：29歳
思考：今度は、29を0.23乗する必要がある。
アクション：
{
  "action": "Calculator",
  "action_input": "29^0.23"
}

観察：答え：2.169459462491557

思考：最終的な答えがわかりました。
最終的な解答：2.169459462491557

> チェーンを終了しました。
'2.169459462491557'
```

## メモリ：チェーンとエージェントに状態を追加する
チャットモデルで初期化されたチェーンとエージェントでメモリを使用することができます。LLMのメモリとの主な違いは、以前のすべてのメッセージを文字列にまとめようとするのではなく、それらを独自のユニークなメモリオブジェクトとして保持できることです。

```python
from langchain.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("以下は、人間とAIのフレンドリーな会話です。 AIはおしゃべりで、コンテキストから具体的な内容をたくさん提供してくれます。 AIは質問の答えを知らない場合、正直に「知らない」と答えます。"),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

llm = ChatOpenAI(temperature=0)
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

conversation.predict(input="こんにちは！")
# -> 'こんにちは！何かお手伝いできることはありますか？'


conversation.predict(input="順調にやってます！ちょうどAIと会話しているところです。")
# -> "それは楽しそうですね！お話しすることができて嬉しいです。何か特定の話題がありますか？"

conversation.predict(input="あなたについて教えてください。")
# -> "もちろんです！私はOpenAIによって作成されたAI言語モデルです。インターネットからの大量のテキストデータセットでトレーニングされているため、人間のような言語を理解し生成することができます。質問に答えたり、情報を提供したり、このような会話をすることさえできます。私について知りたいことが他にありますか？"
```

