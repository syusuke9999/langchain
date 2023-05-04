# クイックスタートガイド

このチュートリアルでは、LangChainを使ってエンドツーエンドの言語モデルアプリケーションを構築する方法について簡単に説明します。

## インストール

はじめに、以下のコマンドでLangChainをインストールします。

```bash
pip install langchain
# または
conda install langchain -c conda-forge
```

## 環境設定

LangChainを使用するには通常、1つ以上のモデルプロバイダ、データストア、APIなどとの統合が必要です。

この例ではOpenAIのAPIを使用するため、まずSDKをインストールする必要があります。

```bash
pip install openai
```

次に、環境変数をターミナルで設定します。

```bash
export OPENAI_API_KEY="..."
```

あるいは、Jupyterノートブック（またはPythonスクリプト）内でこれを行うこともできます。

```python
import os
os.environ["OPENAI_API_KEY"] = "..."
```

## 言語モデルアプリケーションの構築: LLMs

LangChainをインストールし環境を設定したので、言語モデルアプリケーションの構築を始めることができます。

LangChainは、言語モデルアプリケーションを構築するために使用できる多くのモジュールを提供しています。モジュールはより複雑なアプリケーションを作成するために組み合わせることができますし、単純なアプリケーションには個別に使用することができます。

## LLMs: 言語モデルから予測を取得する

LangChainの最も基本的な構成要素は、入力に対してLLMを呼び出すことです。
これを行う簡単な例を見てみましょう。
この目的のために、会社が作るものに基づいて会社名を生成するサービスを構築しているとしましょう。

これを行うために、まずLLMラッパーをインポートする必要があります。

```python
from langchain.llms import OpenAI
```

次に、任意の引数でラッパーを初期化できます。
この例では、出力をもっとランダムにしたいので、高い温度で初期化します。

```python
llm = OpenAI(temperature=0.9)
```

これで、入力に対して呼び出すことができます！

```python
text = "カラフルな靴下を作る会社に良い会社名は何でしょうか？"
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
    template="{product}を作る会社の名前として良いものは何ですか？",
)
```

これがどのように機能するか見てみましょう！`.format`メソッドを使ってフォーマットできます。

```python
print(prompt.format(product="カラフルな靴下"))
```

```pycon
カラフルな靴下を作る会社にふさわしい名前は何ですか？
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
    template="{product}を作る会社の名前として良いものは何ですか？",
)
```

これでユーザー入力を受け取り、それをプロンプトにフォーマットし、LLMに送信する非常にシンプルなチェーンを作成できます。

```python
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)
```

これで製品だけを指定してチェーンを実行できます！

```python
chain.run("カラフルな靴下")
# -> '\n\nSocktastic!'
```

これで最初のチェーン、LLMチェーンができました。
これはより複雑なチェーンを扱う上で基本となるシンプルなタイプのチェーンの一つです。

[詳細については、チェーンの入門ガイドをご覧ください。](../modules/chains/getting_started.ipynb)

## エージェント: ユーザー入力に基づいて動的にチェーンを呼び出す

これまで見てきたチェーンは、あらかじめ決められた順序で実行されます。

エージェントはもはやそうではありません。LLMを使用してどのアクションを実行しどの順序で実行するか決定します。アクションは、ツールを使用してその出力を観察するか、ユーザーに戻るかのいずれかです。

正しく使用するとエージェントは非常に強力になります。このチュートリアルでは最も簡単で最も高レベルなAPIを使って、エージェントを簡単に使う方法を紹介します。

エージェントをロードするには、以下の概念を理解する必要があります。

- ツール: 特定の任務を実行する関数。これにはGoogle検索、データベース検索、Python REPL、他のチェーンなどが含まれます。ツールのインターフェースは、現在入力として文字列を持つことが期待されている関数で、出力として文字列を持つことが期待されています。
- LLM: エージェントを制御する言語モデル。
- エージェント: 使用するエージェント。これはサポートされているエージェントクラスを参照する文字列である必要があります。このノートブックでは最も簡単で最も高レベルなAPIに焦点を当てているため、これは標準のサポートされているエージェントの使用に限定されます。カスタムエージェントを実装したい場合は、カスタムエージェントのドキュメント（近日公開予定）を参照してください。

**エージェント**: サポートされているエージェントとその仕様のリストについては[こちら](../modules/agents/getting_started.ipynb)を参照してください。

**ツール**: 事前定義されたツールとその仕様のリストについては[こちら](../modules/agents/tools/getting_started.md)を参照してください。

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
 まず気温を調べてから、計算機を使って0.023乗する必要があります。
アクション: 検索
アクション入力: "昨日のサンフランシスコの最高気温"
観察: サンフランシスコの昨日の気温。最高気温：57°F（午後1時56分）最低気温：49°F（午前1時56分）平均気温 ...
思考: 気温が分かったので、計算機を使って0.023乗できます。
アクション: 計算機
アクション入力: 57^0.023
観察: 答え: 1.0974509573251117

思考: 最終的な答えが分かりました
最終回答: 昨日のサンフランシスコの最高気温（華氏）を0.023乗した値は1.0974509573251117です。

> チェーンが終了しました。
```


## メモリ: チェーンとエージェントに状態を追加する

これまでのチェーンとエージェントはすべてステートレスでした。しかし、多くの場合、チェーンやエージェントに「メモリ」の概念を持たせて、以前のやり取りに関する情報を覚えておくことが望ましいでしょう。これが最も明確で簡単な例は、チャットボットを設計する場合です。過去のメッセージを覚えておくことで、その文脈を利用してより良い会話ができるようになります。これは「短期記憶」の一種です。もっと複雑な方面では、チェーンやエージェントが時間をかけて重要な情報を覚えておくことができます。これは「長期記憶」の一形態です。後者の具体的なアイデアについては、この[素晴らしい論文](https://memprompt.com/)を参照してください。

LangChainは、この目的のために特別に作成されたいくつかのチェーンを提供しています。このノートブックではそのチェーン（`ConversationChain`）を2つの異なるタイプのメモリで使用する方法を説明します。

デフォルトでは`ConversationChain`は、すべての過去の入力/出力を覚えておいて、渡されるコンテキストに追加する簡単なタイプのメモリを持っています。このチェーンを使ってみましょう（`verbose=True`に設定してプロンプトが表示されるようにします）。

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
以下は、人間とAIのフレンドリーな会話です。AIはおしゃべりで、コンテキストから具体的な内容をたくさん提供してくれます。AIは質問の答えを知らない場合、正直に「知らない」と答えます。

現在の会話:

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
以下は、人間とAIのフレンドリーな会話です。 AIはおしゃべりで、コンテキストから具体的な内容をたくさん提供してくれます。 AIは質問の答えを知らない場合、正直に「知らない」と答えます。

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
    SystemMessage(content="あなたは、英語から日本語に翻訳してくれる役に立つアシスタントです。"),
    HumanMessage(content="I love programming.")
]
chat(messages)
# -> AIMessage(content="J'aime programmer.", additional_kwargs={})
```

さらに一歩進めて`generate`を使って複数のメッセージセットに対して補完を生成することができます。これは追加の`message`パラメータを持つ`LLMResult`を返します。

```python
batch_messages = [
    [
        SystemMessage(content="あなたは、英語から日本語に翻訳してくれる役に立つアシスタントです。"),
        HumanMessage(content="I love programming.")
    ],
    [
        SystemMessage(content="あなたは、英語から日本語に翻訳してくれる役に立つアシスタントです。"),
        HumanMessage(content"I love artificial intelligence.")
    ],
]
result = chat.generate(batch_messages)
result
# -> LLMResult(generations=[[ChatGeneration(text="J'aime programmer.", generation_info=None, message=AIMessage(content="J'aime programmer.", additional_kwargs={}))], [ChatGeneration(text="J'aime l'intelligence artificielle.", generation_info=None, message=AIMessage(content="J'aime l'intelligence artificielle.", additional_kwargs={}))]], llm_output={'token_usage': {'prompt_tokens': 57, 'completion_tokens': 20, 'total_tokens': 77}})
```

このLLMResultからトークン使用量などを回復できます。

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

template = "あなたは、{input_language}を{output_language}に翻訳する便利なアシスタントです。"
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

template = "あなたは、{input_language}を{output_language}に翻訳する便利なアシスタントです。"
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
考え：オリビア・ワイルドのボーイフレンドを検索し、彼の年齢を0.23乗するために電卓を使う必要がある。
アクション：
{
  "action": "Search",
  "action_input": "Olivia Wilde boyfriend"
}

観察：スディケイスとワイルドの関係は2020年11月に終了しました。ワイルドは、2022年のCinemaConでDon't Worry Darlingを発表している間に、子供の親権に関する裁判文書を公に受け取りました。2021年1月、ワイルドはDon't Worry Darlingの撮影中に出会った歌手ハリー・スタイルズと付き合い始めました。
考え：ハリー・スタイルズの現在の年齢を検索する必要がある。
アクション：
{
  "action": "Search",
  "action_input": "Harry Styles age"
}

観察：29歳
考え：今度は、29を0.23乗する必要がある。
アクション：
{
  "action": "Calculator",
  "action_input": "29^0.23"
}

観察：答え：2.169459462491557

考え：最終的な答えがわかりました。
最終的な答え：2.169459462491557

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

