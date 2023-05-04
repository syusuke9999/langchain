# 用語集

これは、LLMアプリケーションの開発時に一般的に使用される用語集です。
概念が初めて紹介された外部の論文や資料への参照や、LangChainでその概念が使用されている場所への参照が含まれています。

## 思考の連鎖プロンプト

モデルに中間的な推論ステップの一連の生成を促すプロンプト技術です。
この挙動を誘発するための簡単な方法として、「ステップ・バイ・ステップで考えてみましょう」とプロンプトに含めることができます。

リソース:

- [Chain-of-Thought 論文](https://arxiv.org/pdf/2201.11903.pdf)
- [Step-by-Step 論文](https://arxiv.org/abs/2112.00114)

## アクションプラン生成

言語モデルを使用してアクションを生成するプロンプトの使用法です。
これらのアクションの結果は、言語モデルにフィードバックされ、次のアクションを生成するために使用されます。

リソース:

- [WebGPT 論文](https://arxiv.org/pdf/2112.09332.pdf)
- [SayCan 論文](https://say-can.github.io/assets/palm_saycan.pdf)

## ReActプロンプト

Chain-of-Thoughtプロンプトとアクションプラン生成を組み合わせたプロンプト技術です。
これにより、モデルはどのアクションを取るべきかを考え、そのアクションを実行するようになります。

リソース:

- [論文](https://arxiv.org/pdf/2210.03629.pdf)
- [LangChain 例](modules/agents/agents/examples/react.ipynb)

## Self-ask

Chain-of-thoughtプロンプトをベースにしたプロンプト方法です。
この方法では、モデルが自分自身にフォローアップの質問を明示的にし、それらの質問に対して外部の検索エンジンが答えを提供します。

リソース:

- [論文](https://ofir.io/self-ask.pdf)
- [LangChain 例](modules/agents/agents/examples/self_ask_with_search.ipynb)

## プロンプトチェーン

複数のLLM呼び出しを組み合わせ、1つのステップの出力が次のステップの入力になるようにすることです。

リソース:

- [PromptChainer 論文](https://arxiv.org/pdf/2203.06566.pdf)
- [Language Model Cascades](https://arxiv.org/abs/2207.10342)
- [ICE Primer Book](https://primer.ought.org/)
- [Socratic Models](https://socraticmodels.github.io/)

## メメティックプロキシ

モデルが特定の方法で応答するように促すために、モデルが知っているコンテキストで議論をフレーム化し、そのタイプの応答が得られるようにすることです。例えば、生徒と先生の間の会話としてです。

リソース:

- [論文](https://arxiv.org/pdf/2102.07350.pdf)

## 自己整合性

多様な推論パスをサンプリングし、最も整合性のある回答を選択するデコーディング戦略です。
Chain-of-thoughtプロンプトと組み合わせると効果的です。

リソース:

- [論文](https://arxiv.org/pdf/2203.11171.pdf)

## インセプション

「ファーストパーソンインストラクション」とも呼ばれます。
プロンプトにモデルの応答の開始部分を含めることで、モデルが特定の方法で考えるように促します。

リソース:

- [例](https://twitter.com/goodside/status/1583262455207460865?s=20&t=8Hz7XBnK1OF8siQrxxCIGQ)

## MemPrompt

MemPromptは、エラーやユーザーフィードバックのメモリを維持し、それらを使用してミスの繰り返しを防ぎます。

リソース:

- [論文](https://memprompt.com/)