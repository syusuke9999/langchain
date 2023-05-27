# クイックスタートガイド


このチュートリアルではLangChainを使ったエンドツーエンドの言語モデルアプリケーションの構築について、簡単なウォークスルーを紹介します。

## インストール

まずは以下のコマンドでLangChainをインストールします:

```bash
pip install langchain
# or
conda install langchain -c conda-forge
```


## 環境設定

LangChainを使用するには通常1つ以上のモデルプロバイダー、データストア、APIなどとの統合が必要です。

今回のサンプルではOpenAIのAPIを使用するので、最初にOpenAIのSDKをインストールする必要があります:

```bash
pip install openai
```

次に、ターミナルで環境変数を設定する必要があります。

```bash
export OPENAI_API_KEY="..."
```

代わりに、Jupyterノートブック（またはPythonスクリプト）の内部から行うこともできます：

```python
import os
os.environ["OPENAI_API_KEY"] = "..."
```

If you want to set the API key dynamically, you can use the openai_api_key parameter when initiating OpenAI class—for instance, each user's API key.

```python
from langchain.llms import OpenAI
llm = OpenAI(openai_api_key="OPENAI_API_KEY")
```

## 言語モデルアプリケーションの構築：LLMs

LangChainをインストールし、環境を整えたところで、言語モデルアプリケーションのビルドを開始します。

LangChainは、言語モデルアプリケーションをビルドするために使用できる多くのモジュールを提供しています。モジュールを組み合わせてより複雑なアプリケーションを作成することもできますし、個別に使用してシンプルなアプリケーションを作成することもできます。



## LLMs：言語モデルから予測を取得する

LangChainの最も基本的な構成要素は、ある入力に対してLLMを呼び出すことです。
では、その方法を簡単なサンプルで説明しましょう。
ここでは、例えば、会社が作っているものに基づいて会社名を生成するサービスを作ることにしましょう。

そのために、まずLLMラッパーをインポートする必要があります。

```python
from langchain.llms import OpenAI
```

そして任意の引数でラッパーを初期化することができます。
この例では出力をよりランダムにしたいでしょうから、高い温度で初期化することにします。

```python
llm = OpenAI(temperature=0.9)
```

これで、入力に対して呼び出すことができます！

```python
text = "カラフルな靴下を作る会社にとって、良い会社名は何でしょうか？"
print(llm(text))
```

```pycon
Feetful of Fun
```

LangChain内でLLMsを使用する方法の詳細については[LLM入門ガイド](../modules/models/llms/getting_started.ipynb)を参照してください。


## プロンプトテンプレート:LLMsのプロンプトを管理する

LLMを呼び出すことは素晴らしい最初のステップですが、これはほんの始まりに過ぎません。
通常アプリケーションでLLMを利用する場合、ユーザー入力を直接LLMに送信することはありません。
その代わりにユーザーの入力を受けてプロンプトを作成し、それをLLMに送るということが多いのではないでしょうか。

例えば先ほどの例では、カラフルな靴下を作る会社の名前を尋ねるテキストがハードコーディングされていました。
この架空のサービスでは会社の事業内容を記述するユーザーの入力だけを受け取り、その情報を使ってプロンプトをフォーマットしたいと思うでしょう。

LangChainを使えば簡単にできます！

まず、プロンプトテンプレートを定義しましょう：

```python
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["product"],
    template="{product}を作っている会社の名前として良いものは何ですか？{product}?",
)
```

ではこの動作を確認してみましょう！`.format`メソッドを呼び出してフォーマットすることができます。

```python
print(prompt.format(product="カラフルな靴下"))
```

```pycon
カラフルな靴下を作っている会社の名前として良いものは何ですか？
```


[詳細については、プロンプトのスタートガイドをご覧ください](../modules/prompts/chat_prompt_template.ipynb)




## チェーン：マルチステップのワークフローでLLMとプロンプトを組み合わせる。

これまでは、PromptTemplateとLLmのプリミティブを単体で扱ってきました。しかし、もちろん実際のアプリケーションは1つのプリミティブだけでなく、それらを組み合わせて使うことになります。

LangChainのチェーンはリンクで構成さており、リンクはLLMのようなプリミティブでも他のチェーンでも構いません。

チェーンの最も核となるタイプはLLMChainで、PromptTemplateとLLMで構成されています。

先ほどのサンプルを拡張すると、ユーザーの入力を受けてPromptTemplateで整形し、整形された応答をLLMに渡すLLMChainを組み立てることができます。

```python
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["product"],
    template="{product}を作る会社にとって、良い名前は何でしょうか？",
)
```

これでユーザーの入力を受け取り、それを使ってプロンプトをフォーマットして、LLMに送信するという非常にシンプルなチェーンを作れるようになりました:

```python
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)
```

これで、製品を指定するだけでそのチェーンを実行できるようになりました！

```python
chain.run("colorful socks")
# -> '\n\nSocktastic!'
```

さあ最初のチェーン、LLMチェーンができました。
これはチェーンの中でも比較的単純なタイプですが、その仕組みを理解することで、より複雑なチェーンに対応できるようになります。

[詳細については、チェーンのスタートガイドをご覧ください](../modules/chains/getting_started.ipynb)

## エージェント:ユーザー入力に基づいてチェーンを動的に呼び出す

これまで見てきたチェーンはあらかじめ決められた順序で実行されるものでした。

しかしエージェントはそうではありません。エージェントはLLMを使って、どのアクションをどの順番で行うかを決定します。アクションはツールを使ってその出力を観察したり、ユーザーに戻ったりすることができます。

正しく使用すれば、エージェントは非常に強力なものになります。このチュートリアルでは、最もシンプルでハイレベルなAPIを使用して、エージェントを簡単に使用する方法を示します。


エージェントを読み込むためには、以下の概念を理解する必要があります。:

- ツール：特定の任務を実行する関数。これにはGoogle検索、データベース検索、Python REPLや他のチェーンなどが含まれます。ツールのインターフェースは、現在入力として文字列を持つことが期待される関数で、出力も文字列です。
- LLM：エージェントを動作させる言語モデル。
- エージェント：使用するエージェント。これはサポートされているエージェントクラスを参照する文字列である必要があります。このノートブックは最もシンプルでハイレベルなAPIに焦点を当てているため、標準でサポートされているエージェントの使用についてのみカバーしています。カスタムエージェントを実装したい場合は、カスタムエージェントのドキュメント（近日公開予定）を参照してください。

**エージェント**：サポートされているエージェントとその仕様のリストについては、[こちら](../modules/agents/getting_started.ipynb)を参照してください。

**ツール**：事前定義されたツールとその仕様のリストについては、[こちら](../modules/agents/tools/getting_started.md)を参照してください。

この例では、SerpAPI Pythonパッケージもインストールする必要があります。

```bash
pip install google-search-results
```

そして、適切な環境変数を設定してください。

```python
import os
os.environ["SERPAPI_API_KEY"] = "..."
```

それでは始めましょう！

```python
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

# まず、エージェントを制御するために使用する言語モデルをロードしましょう。
llm = OpenAI(temperature=0)

# 次に、使用するツールをいくつかロードしましょう。`llm-math`ツールはLLMを使用するため、それを渡す必要があります。
tools = load_tools(["serpapi", "llm-math"], llm=llm)


# 最後に、ツール、言語モデル、使用したいエージェントのタイプを使ってエージェントを初期化しましょう。
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# さあ、試してみましょう！
agent.run("昨日のサンフランシスコの最高気温は摂氏で何度でしたか？その数値を0.023乗したらどうなりますか？")
```

```pycon
> Entering new AgentExecutor chain...
 I need to find the temperature first, then use the calculator to raise it to the .023 power.
Action: Search
Action Input: "High temperature in SF yesterday"
Observation: San Francisco Temperature Yesterday. Maximum temperature yesterday: 57 °F (at 1:56 pm) Minimum temperature yesterday: 49 °F (at 1:56 am) Average temperature ...
Thought: I now have the temperature, so I can use the calculator to raise it to the .023 power.
Action: Calculator
Action Input: 57^.023
Observation: Answer: 1.0974509573251117

Thought: I now know the final answer
Final Answer: The high temperature in SF yesterday in Fahrenheit raised to the .023 power is 1.0974509573251117.

> Finished chain.
```



## メモリ：チェーンとエージェントにステートを追加する。

これまでに紹介したチェーンやエージェントはすべてステートレス（状態を持たない）でした。しかしチェーンやエージェントになんらかの「記憶」の概念を持たせて、以前の対話に関する情報を記憶させたいと思うことがよくあります。最も明確でシンプルな例としては、チャットボットを設計する際に以前のメッセージを記憶させ、そこから得られる文脈に基づいてより良い会話ができるようにしたいと考えます。これは「短期記憶」の一種と言えるでしょう。もっと複雑な例としてはチェーンやエージェントが重要な情報を長期的に記憶していくようなことも考えられます-これは「長期記憶」の一種と言えるかもしれません。後者に関するより具体的なアイデアについてはこの[素晴らしい論文](https://memprompt.com/)を参照してください。

LangChainはこの目的のために特別に作られたチェーンをいくつか提供しています。このノートブックではそのうちの1つのチェーン（`ConversationChain`）を2つの異なるタイプのメモリで使用する方法を説明します。

デフォルトでは`ConversationChain`は単純なタイプのメモリを持っており、以前の入出力をすべて記憶して渡されたコンテキストに追加します。このチェーンの使い方を見てみましょう（プロンプトが表示されるように`verbose=True`を設定しています）。

```python
from langchain import OpenAI, ConversationChain

llm = OpenAI(temperature=0)
conversation = ConversationChain(llm=llm, verbose=True)

output = conversation.predict(input="Hi there!")
print(output)
```

```pycon
> Entering new chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:

Human: Hi there!
AI:

> Finished chain.
' Hello! How are you today?'
```

```python
output = conversation.predict(input="順調にやってます！AIと会話しているだけ。")
print(output)
```

```pycon
> Entering new chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:

Human: こんにちは！
AI:  こんにちは！今日はいかがお過ごしですか？
Human: I'm doing well! Just having 順調にやってます！AIと会話しているだけ。a conversation with an AI.
AI:

> Finished chain.
" それは素晴らしいですね！どんな話をしたいですか？"
```

## 言語モデルアプリケーションの構築：チャットモデル

同様に、LLMの代わりにチャットモデルを使用することもできます。チャットモデルは、言語モデルの一種です。
チャットモデルは内部で言語モデルを使用していますが、公開されるインターフェースは少し異なります。すなわち「テキスト入力、テキスト出力」のAPIを公開するのではなく、チャットメッセージ」が入力と出力となるインターフェースを公開します。

チャットモデルAPIは比較的新しいため、適切な抽象化をまだ見つけている最中です。

## チャットモデルからメッセージ補完を取得する

チャットモデルに1つ以上のメッセージを渡すことで、チャット補完を取得できます。返されるのはメッセージです。LangChainで現在サポートされているメッセージの種類は、`AIMessage`、`HumanMessage`、`SystemMessage`、および`ChatMessage`です。`ChatMessage`は任意の役割パラメータを受け取ります。ほとんどの場合、`HumanMessage`、`AIMessage`、および`SystemMessage`を扱うことになります。

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

chat = ChatOpenAI(temperature=0)
```

1つのメッセージを渡すことで、補完を取得することができます。

```python
chat([HumanMessage(content="この文章を英語から日本語に訳してください。私はプログラミングが大好きです。")])
# -> AIMessage(content="J'aime programmer.", additional_kwargs={})
```

また、OpenAIのgpt-3.5-turboおよびgpt-4モデルに複数のメッセージを渡すこともできます。

```python
messages = [
    SystemMessage(content="あなたは、英語を日本語に翻訳してくれる役に立つアシスタントです。"),
    HumanMessage(content="I love programming.")
]
chat(messages)
# -> AIMessage(content="J'aime programmer.", additional_kwargs={})
```

さらに一歩進めて`generate`を使用して複数のメッセージセットに対して補完を生成することができます。これにより、追加の`message`パラメータを持つ`LLMResult`が返されます。
```python
batch_messages = [
    [
        SystemMessage(content="あなたは、英語を日本語に翻訳してくれる役に立つアシスタントです。"),
        HumanMessage(content="I love programming.")
    ],
    [
        SystemMessage(content="あなたは、英語を日本語に翻訳してくれる役に立つアシスタントです。"),
        HumanMessage(content="I love artificial intelligence.")
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
LLMと同様に`MessagePromptTemplate`を使うことでテンプレート化を利用することができます。1つまたは複数の`MessagePromptTemplate`から`ChatPromptTemplate`を作成することができます。`ChatPromptTemplate`の`format_prompt`--これは`PromptValue`を返しフォーマットされた値をllmやチャットモデルの入力として利用したいのかどうかに応じて、文字列または`Message`オブジェクトに変換することができます。

便宜上テンプレートには`from_template`メソッドが公開されています。もしこのテンプレートを使うとしたら以下のようになります：

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

chat = ChatOpenAI(temperature=0)

template = "あなたは、{input_language}を{output_language}に翻訳してくれる役に立つアシスタントです。"
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# フォーマットされたメッセージからチャット補完を取得する
chat(chat_prompt.format_prompt(input_language="English", output_language="Japnese", text="I love programming.").to_messages())
# -> AIMessage(content="J'aime programmer.", additional_kwargs={})
```

## チャットモデルによるチェーン
上記のセクションで説明した`LLMChain`はチャットモデルでも同様に使用することができます：

```python
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

chat = ChatOpenAI(temperature=0)

template = "あなたは、{input_language}を{output_language}に翻訳してくれる役立つアシスタントです。"
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

chain = LLMChain(llm=chat, prompt=chat_prompt)
chain.run(input_language="English", output_language="Japanese", text="I love programming.")
# -> "J'aime programmer."
```

## チャットモデルを使用したエージェント
エージェントはチャットモデルとも使用できます。エージェントタイプとして`AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION`を使用して初期化できます。

```python
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

# まず、エージェントをコントロールするために使う言語モデルを読み込みましょう。
chat = ChatOpenAI(temperature=0)

# 次に、使用するツールをいくつかロードしましょう。なお、`llm-math`ツールはLLMを使用するので、それを渡す必要があります。
llm = OpenAI(temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)


# 最後に、ツール、言語モデル、そして使いたいタイプのエージェントを初期化してみましょう。
agent = initialize_agent(tools, chat, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# それでは、テストしてみましょう！
agent.run("オリビア・ワイルドのボーイフレンドは誰？彼の現在の年齢を0.23乗するといくつになりますか？")
```

```pycon

> Entering new AgentExecutor chain...
Thought: I need to use a search engine to find Olivia Wilde's boyfriend and a calculator to raise his age to the 0.23 power.
Action:
{
  "action": "Search",
  "action_input": "Olivia Wilde boyfriend"
}

Observation: Sudeikis and Wilde's relationship ended in November 2020. Wilde was publicly served with court documents regarding child custody while she was presenting Don't Worry Darling at CinemaCon 2022. In January 2021, Wilde began dating singer Harry Styles after meeting during the filming of Don't Worry Darling.
Thought:I need to use a search engine to find Harry Styles' current age.
Action:
{
  "action": "Search",
  "action_input": "Harry Styles age"
}

Observation: 29 years
Thought:Now I need to calculate 29 raised to the 0.23 power.
Action:
{
  "action": "Calculator",
  "action_input": "29^0.23"
}

Observation: Answer: 2.169459462491557

Thought:I now know the final answer.
Final Answer: 2.169459462491557

> Finished chain.
'2.169459462491557'
```
## メモリ：チェーンとエージェントに状態を持たせる
チャットモデルで初期化されたチェーンやエージェントでMemoryを使うことができます。LLMのMemoryとの主な違いは、過去のすべてのメッセージを文字列に集約しようとするのではなく、独自のユニークなMemoryオブジェクトとして保持できる点です。

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
    SystemMessagePromptTemplate.from_template("これは、人間とAIのフレンドリーな会話です。AIはおしゃべりで、コンテキストから具体的な詳細をたくさん提供してくれます。AIは質問の答えを知らない場合、正直に「知りません」と答えます。"),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

llm = ChatOpenAI(temperature=0)
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

conversation.predict(input="こんにちは！")
# -> 'こんにちは！今日はどのようなご用件でしょうか？'


conversation.predict(input="順調にやってます！AIと会話しているだけ。")
# -> "それは楽しそうですね！一緒におしゃべりできるのがうれしいです。何か具体的に話したいことはありますか？"

conversation.predict(input="あなた自身のことを教えてください。")
# -> "もちろんです！私はOpenAIによって作られたAI言語モデルです。インターネット上の大規模なテキストデータセットでトレーニングされたので、人間のように言語を理解し生成することができます。質問に答えたり、情報を提供したり、こんな感じで会話することもできます。他に私について知りたいことはありますか？you'd like to know about me?"
```

