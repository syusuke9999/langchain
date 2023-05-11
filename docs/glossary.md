# 用語集

これは、LLMアプリケーションを開発する際に一般的に使用される用語の集まりです。
概念が初めて紹介された外部の論文や資料への参照や、LangChainでその概念が使用されている場所への参照が含まれています。

## 思考の連鎖プロンプト

モデルに中間的な推論ステップの一連の生成を促すプロンプト技術。
この挙動を誘発するための簡単な方法として、プロンプトに「ステップ・バイ・ステップで考えてみましょう」と含めることができます。

リソース:

- [Chain-of-Thought Paper](https://arxiv.org/pdf/2201.11903.pdf)
- [Step-by-Step Paper](https://arxiv.org/abs/2112.00114)

## アクションプラン生成

言語モデルを使用してアクションを生成するプロンプトの使用法。
これらのアクションの結果は、言語モデルにフィードバックされ、次のアクションを生成するために使用されます。

リソース:

- [WebGPT Paper](https://arxiv.org/pdf/2112.09332.pdf)
- [SayCan Paper](https://say-can.github.io/assets/palm_saycan.pdf)

## ReActプロンプト

思考の連鎖プロンプトとアクションプラン生成を組み合わせたプロンプト技術。
これにより、モデルはどのアクションを取るべきかを考え、そのアクションを実行するようになります。

リソース:

- [Paper](https://arxiv.org/pdf/2210.03629.pdf)
- [LangChain Example](modules/agents/agents/examples/react.ipynb)

## Self-ask

思考の連鎖プロンプトをベースにしたプロンプト方法。
この方法では、モデルが自分自身にフォローアップの質問を明示的にし、それらの質問には外部の検索エンジンが答える。

リソース:

- [Paper](https://ofir.io/self-ask.pdf)
- [LangChain Example](modules/agents/agents/examples/self_ask_with_search.ipynb)

## プロンプトチェーン

複数のLLM呼び出しを組み合わせ、1つのステップの出力が次のステップの入力になるようにする。

リソース:

- [PromptChainer Paper](https://arxiv.org/pdf/2203.06566.pdf)
- [Language Model Cascades](https://arxiv.org/abs/2207.10342)
- [ICE Primer Book](https://primer.ought.org/)
- [Socratic Models](https://socraticmodels.github.io/)

## メメティックプロキシ

LLMに特定の方法で応答するように促すために、モデルが知っているコンテキストで議論をフレームし、そのタイプの応答が得られるようにする。例えば、生徒と先生の間の会話として。

リソース:

- [Paper](https://arxiv.org/pdf/2102.07350.pdf)

## 自己整合性

多様な推論パスのサンプルを取り、最も整合性のある回答を選択するデコーディング戦略。
思考の連鎖プロンプトと組み合わせることで最も効果的です。

リソース:

- [Paper](https://arxiv.org/pdf/2203.11171.pdf)

## インセプション

「ファーストパーソンインストラクション」とも呼ばれます。
プロンプトにモデルの応答の開始部分を含めることで、モデルが特定の方法で考えるように促します。

リソース:

- [Example](https://twitter.com/goodside/status/1583262455207460865?s=20&t=8Hz7XBnK1OF8siQrxxCIGQ)

## MemPrompt

MemPromptは、エラーやユーザーフィードバックのメモリを維持し、それらを使用してミスの繰り返しを防ぎます。

リソース:

- [Paper](https://memprompt.com/)
