# 🦜️🔗 LangChain

⚡ LLMを使ったアプリケーション構築のための組み合わせ性 ⚡

[![lint](https://github.com/hwchase17/langchain/actions/workflows/lint.yml/badge.svg)](https://github.com/hwchase17/langchain/actions/workflows/lint.yml)
[![test](https://github.com/hwchase17/langchain/actions/workflows/test.yml/badge.svg)](https://github.com/hwchase17/langchain/actions/workflows/test.yml)
[![linkcheck](https://github.com/hwchase17/langchain/actions/workflows/linkcheck.yml/badge.svg)](https://github.com/hwchase17/langchain/actions/workflows/linkcheck.yml)
[![Downloads](https://static.pepy.tech/badge/langchain/month)](https://pepy.tech/project/langchain)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchainai.svg?style=social&label=Follow%20%40LangChainAI)](https://twitter.com/langchainai)
[![](https://dcbadge.vercel.app/api/server/6adMQxSpJS?compact=true&style=flat)](https://discord.gg/6adMQxSpJS)
[![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/hwchase17/langchain)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/hwchase17/langchain)
[![GitHub star chart](https://img.shields.io/github/stars/hwchase17/langchain?style=social)](https://star-history.com/#hwchase17/langchain)
[![Dependency Status](https://img.shields.io/librariesio/github/hwchase17/langchain)](https://libraries.io/github/hwchase17/langchain)
[![Open Issues](https://img.shields.io/github/issues-raw/hwchase17/langchain)](https://github.com/hwchase17/langchain/issues)


JS/TS版をお探しですか？ [LangChain.js](https://github.com/hwchase17/langchainjs)をチェックしてください。

**本番サポート:** LangChainsを本番環境に移行する際に、より包括的なサポートを提供したいと考えています。
[このフォーム](https://forms.gle/57d8AmXBYp8PP8tZA)に記入していただくと、専用のサポートSlackチャンネルを設定します。

## クイックインストール

`pip install langchain`
または
`conda install langchain -c conda-forge`

## 🤔 これは何？

大規模言語モデル（LLM）は、開発者がこれまで構築できなかったアプリケーションを構築することを可能にする画期的な技術として登場しています。しかし、これらのLLMを単独で使用するだけでは、本当に強力なアプリを作成するには不十分であり、他の計算や知識のソースと組み合わせることができるときに真の力が発揮されます。

このライブラリは、そのようなアプリケーションの開発を支援することを目的としています。これらのアプリケーションの一般的な例は以下の通りです。

**❓ 特定のドキュメントに対する質問回答**

- [ドキュメント](https://langchain.readthedocs.io/en/latest/use_cases/question_answering.html)
- エンドツーエンドの例: [Notionデータベースに対する質問回答](https://github.com/hwchase17/notion-qa)

**💬 チャットボット**

- [ドキュメント](https://langchain.readthedocs.io/en/latest/use_cases/chatbots.html)
- エンドツーエンドの例: [Chat-LangChain](https://github.com/hwchase17/chat-langchain)

**🤖 エージェント**

- [ドキュメント](https://langchain.readthedocs.io/en/latest/modules/agents.html)
- エンドツーエンドの例: [GPT+WolframAlpha](https://huggingface.co/spaces/JavaFXpert/Chat-GPT-LangChain)

## 📖 ドキュメント

以下の内容についての詳細なドキュメントは[こちら](https://langchain.readthedocs.io/en/latest/?)をご覧ください。

- はじめに（インストール、環境設定、簡単な例）
- How-Toの例（デモ、統合、ヘルパー関数）
- リファレンス（完全なAPIドキュメント）
- リソース（コア概念の高レベル説明）

## 🚀 これは何に役立ちますか？

LangChainは主に6つの領域で役立つように設計されています。
これらは、複雑さの点で次のような順序です。

**📃 LLMとプロンプト:**

プロンプト管理、プロンプト最適化、すべてのLLMのための一般的なインターフェース、およびLLMを操作するための一般的なユーティリティが含まれます。

**🔗 チェーン:**

チェーンは単一のLLM呼び出しを超えて、シーケンス内の呼び出し（LLMまたは別のユーティリティへのもの）を含みます。LangChainは、チェーン用の標準インターフェース、他のツールとの多くの統合、および一般的なアプリケーション向けのエンドツーエンドチェーンを提供します。

**📚 データ拡張生成:**

データ拡張生成は、外部データソースとやり取りして生成ステップで使用するデータを取得する最初のタイプのチェーンを含みます。例としては、長いテキストの要約や特定のデータソースに対する質問/回答があります。

**🤖 エージェント:**

エージェントは、LLMがどのアクションを選択し、そのアクションを実行し、観測を見て、終了するまで繰り返すことを含みます。LangChainは、エージェント用の標準インターフェース、選択可能なエージェントのセット、およびエンドツーエンドのエージェントの例を提供します。

**🧠 メモリ:**

メモリは、チェーン/エージェントの呼び出し間で状態を維持することを指します。LangChainは、メモリ用の標準インターフェース、メモリ実装のコレクション、およびメモリを使用するチェーン/エージェントの例を提供します。

**🧐 評価:**

[BETA] 生成モデルは、従来の指標で評価するのが非常に難しいことで知られています。新しい評価方法の1つは、言語モデル自体を使用して評価を行うことです。LangChainは、これを支援するためのプロンプト/チェーンを提供します。

これらの概念に関する詳細は、[完全なドキュメント](https://langchain.readthedocs.io/en/latest/)を参照してください。

## 💁 貢献

急速に発展する分野のオープンソースプロジェクトとして、新機能の追加、インフラの改善、ドキュメントの向上など、あらゆる形での貢献を大歓迎しています。

貢献方法の詳細については、[こちら](.github/CONTRIBUTING.md)を参照してください。

