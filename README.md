# 🦜️🔗 LangChain

⚡ LLMを使ったアプリケーション構築を通じた組み合わせ可能性 ⚡

[![lint](https://github.com/hwchase17/langchain/actions/workflows/lint.yml/badge.svg)](https://github.com/hwchase17/langchain/actions/workflows/lint.yml) [![test](https://github.com/hwchase17/langchain/actions/workflows/test.yml/badge.svg)](https://github.com/hwchase17/langchain/actions/workflows/test.yml) [![linkcheck](https://github.com/hwchase17/langchain/actions/workflows/linkcheck.yml/badge.svg)](https://github.com/hwchase17/langchain/actions/workflows/linkcheck.yml) [![Downloads](https://static.pepy.tech/badge/langchain/month)](https://pepy.tech/project/langchain) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchainai.svg?style=social&label=Follow%20%40LangChainAI)](https://twitter.com/langchainai) [![](https://dcbadge.vercel.app/api/server/6adMQxSpJS?compact=true&style=flat)](https://discord.gg/6adMQxSpJS) [![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/hwchase17/langchain) [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/hwchase17/langchain)


JS/TSバージョンをお探しですか？ [LangChain.js](https://github.com/hwchase17/langchainjs)をチェックしてください。

**本番環境サポート:** LangChainsを本番環境に移行する際に、より包括的なサポートを提供したいと考えています。
[このフォーム](https://forms.gle/57d8AmXBYp8PP8tZA)に記入していただくと、専用のサポートSlackチャンネルを設定します。

## クイックインストール

`pip install langchain`
または
`conda install langchain -c conda-forge`

## 🤔 これは何？

大規模言語モデル（LLMs）は、開発者がこれまでできなかったアプリケーションを構築できるようにする、革新的なテクノロジーとして注目されています。 しかし、これらのLLMsを単独で使っても真にパワフルなアプリケーションを作るには不十分な場合が多く、LLMsを他の計算ソースや知識と組み合わせることで真のパワーを発揮することができます。

このライブラリーはそのようなアプリケーションの開発を支援することを目的としています。これらのアプリケーションの一般的なサンプルは以下の通りです：

**❓ 特定のドキュメントに対する質問と応答**

- [ドキュメント](https://langchain.readthedocs.io/en/latest/use_cases/question_answering.html)
- エンドツーエンドの例: [Notionデータベースに対する質問回答](https://github.com/hwchase17/notion-qa)

**💬 チャットボット**

- [ドキュメント](https://langchain.readthedocs.io/en/latest/use_cases/chatbots.html)
- エンドツーエンドの例: [Chat-LangChain](https://github.com/hwchase17/chat-langchain)

**🤖 エージェント**

- [ドキュメント](https://langchain.readthedocs.io/en/latest/modules/agents.html)
- エンドツーエンドの例: [GPT+WolframAlpha](https://huggingface.co/spaces/JavaFXpert/Chat-GPT-LangChain)

## 📖 ドキュメント

以下の内容についての詳細なドキュメントは[こちら](https://langchain.readthedocs.io/en/latest/?)を参照してください。

- はじめに（インストール、環境設定、簡単な例）
- How-Toの例（デモ、統合、ヘルパー関数）
- リファレンス（完全なAPIドキュメント）
- リソース（コアコンセプトの高レベルな説明）

## 🚀 これは何に役立ちますか？

LangChainは、主に6つの領域で役立つように設計されています。
これらは、複雑さの増す順に以下の通りです。

**📃 LLMsとプロンプト:**

これにはプロンプト管理、プロンプト最適化、すべてのLLMsのための汎用のインターフェース、LLMsを操作するための共通のユーティリティが含まれます。

**🔗 チェーン:**

チェーンは単一のLLM呼び出しを越えて、（LLMや別のユーティリティへの）呼び出しのシーケンスを伴います。LangChainは、チェーンのための標準インターフェース、他のツールとの豊富なインテグレーション、そして一般的なアプリケーション向けのエンド・ツー・エンドのチェーンなどを提供します。

**📚 データ増強生成：**

データ増強生成では、最初に外部データソースと対話をし、生成ステップで使用するデータを取得する、特定のタイプのチェーンが含まれます。 サンプルとして、長いテキストの要約や、特定のデータソースに対する質問/回答などがあります。

**🤖 エージェント:**

エージェントはLLMがどのアクションを取るか決断を下し、そのアクションを実行し観察結果を確認し、完了するまでそれを繰り返すというものです。LangChainはエージェントの標準インターフェース、エージェントの選択、エージェントのエンドツーエンドのサンプルを提供しています。

**🧠 メモリ:**

メモリは、チェーン/エージェントの呼び出し間で状態を維持することを指します。LangChainは、メモリの標準インターフェース、メモリ実装のコレクション、およびメモリを使用するチェーン/エージェントの例を提供します。

**🧐 評価:**

[BETA] 生成モデルは、従来の指標で評価するのが非常に難しいことで知られています。それらを評価する新しい方法の1つは、言語モデル自体を使用して評価を行うことです。LangChainは、これを支援するためのいくつかのプロンプト/チェーンを提供します。

これらの概念の詳細については、[完全なドキュメント](https://langchain.readthedocs.io/en/latest/)を参照してください。

## 💁 貢献する方法

急速に発展している分野のオープンソースプロジェクトとして、新機能の追加、インフラの改善、ドキュメントの向上など、どのような形での貢献も大歓迎です。

貢献方法の詳細については、[こちら](.github/CONTRIBUTING.md)を参照してください。
