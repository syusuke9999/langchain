"""Wrapper around FAISS vector database."""
from __future__ import annotations

import math
import pickle
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from langchain.docstore.base import AddableMixin, Docstore
from langchain.docstore.document import Document
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.utils import maximal_marginal_relevance


def dependable_faiss_import() -> Any:
    """Import faiss if available, otherwise raise error."""
    try:
        import faiss
    except ImportError:
        raise ValueError(
            "Could not import faiss python package. "
            "Please install it with `pip install faiss` "
            "or `pip install faiss-cpu` (depending on Python version)."
        )
    return faiss


def _default_relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    # The 'correct' relevance function
    # may differ depending on a few things, including:
    # - the distance / similarity metric used by the VectorStore
    # - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
    # - embedding dimensionality
    # - etc.
    # This function converts the euclidean norm of normalized embeddings
    # (0 is most similar, sqrt(2) most dissimilar)
    # to a similarity function (0 to 1)
    return 1.0 - score / math.sqrt(2)


class FAISS(VectorStore):
    """FAISSベクトルデータベースをラップするクラス。

        使い方: ``faiss`` pythonパッケージがインストールされていること。

        例:
            .. code-block:: python

                from langchain import FAISS
                faiss = FAISS(embedding_function, index, docstore, index_to_docstore_id)

    """

    def __init__(
        self,
        embedding_function: Callable,
        index: Any,
        docstore: Docstore,
        index_to_docstore_id: Dict[int, str],
        relevance_score_fn: Optional[
            Callable[[float], float]
        ] = _default_relevance_score_fn,
    ):
        """Initialize with necessary components."""
        self.embedding_function = embedding_function
        self.index = index
        self.docstore = docstore
        self.index_to_docstore_id = index_to_docstore_id
        self.relevance_score_fn = relevance_score_fn

    def __add(
        self,
        texts: Iterable[str],
        embeddings: Iterable[List[float]],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        if not isinstance(self.docstore, AddableMixin):
            raise ValueError(
                "If trying to add texts, the underlying docstore should support "
                f"adding items, which {self.docstore} does not"
            )
        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            documents.append(Document(page_content=text, metadata=metadata))
        # Add to the index, the index_to_id mapping, and the docstore.
        starting_len = len(self.index_to_docstore_id)
        self.index.add(np.array(embeddings, dtype=np.float32))
        # Get list of index, id, and docs.
        full_info = [
            (starting_len + i, str(uuid.uuid4()), doc)
            for i, doc in enumerate(documents)
        ]
        # Add information to docstore and index.
        self.docstore.add({_id: doc for _, _id, doc in full_info})
        index_to_id = {index: _id for index, _id, _ in full_info}
        self.index_to_docstore_id.update(index_to_id)
        return [_id for _, _id, _ in full_info]

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """ベクトルストアにさらにテキストを埋め込み、追加します。
                引数:
                    texts: ベクトルストアに追加する文字列のイテラブル。
                    metadata: テキストに関連付けられたメタデータのオプションのリスト。
                戻り値:
                    ベクトルストアにテキストを追加することで得られるIDのリスト。
        """
        if not isinstance(self.docstore, AddableMixin):
            raise ValueError(
                "If trying to add texts, the underlying docstore should support "
                f"adding items, which {self.docstore} does not"
            )
        # Embed and create the documents.
        embeddings = [self.embedding_function(text) for text in texts]
        return self.__add(texts, embeddings, metadatas, **kwargs)

    def add_embeddings(
        self,
        text_embeddings: Iterable[Tuple[str, List[float]]],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """ベクトルストアに追加するために、より多くのテキストを埋め込みに通す。

                引数:
                    text_embeddings: ベクトルストアに追加するための文字列と埋め込みのイテラブルなペア。
                    metadata: テキストに関連付けられたメタデータのオプションのリスト。

                戻り値:
                    ベクトルストアにテキストを追加することで得られるIDのリスト。
        """
        if not isinstance(self.docstore, AddableMixin):
            raise ValueError(
                "If trying to add texts, the underlying docstore should support "
                f"adding items, which {self.docstore} does not"
            )
        # Embed and create the documents.

        texts = [te[0] for te in text_embeddings]
        embeddings = [te[1] for te in text_embeddings]
        return self.__add(texts, embeddings, metadatas, **kwargs)

    def similarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 4
    ) -> List[Tuple[Document, float]]:
        """クエリに最も類似したドキュメントを返す。

                引数:
                    query: 類似したドキュメントを検索するためのテキスト。
                    k: 返すドキュメントの数。デフォルトは4。

                戻り値:
                    クエリに最も類似したドキュメントのリストとそれぞれのスコア
        """
        scores, indices = self.index.search(np.array([embedding], dtype=np.float32), k)
        docs = []
        for j, i in enumerate(indices[0]):
            if i == -1:
                # This happens when not enough docs are returned.
                continue
            _id = self.index_to_docstore_id[i]
            doc = self.docstore.search(_id)
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            docs.append((doc, scores[0][j]))
        return docs

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        """クエリに最も類似したドキュメントを返す。

                引数:
                    query: 類似したドキュメントを検索するためのテキスト。
                    k: 返すドキュメントの数。デフォルトは4。

                戻り値:
                    クエリに最も類似したドキュメントのリストとそれぞれのスコア
        """
        embedding = self.embedding_function(query)
        docs = self.similarity_search_with_score_by_vector(embedding, k)
        return docs

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """埋め込みベクトルに最も類似したドキュメントを返します。

                引数:
                    embedding: 類似したドキュメントを検索するための埋め込み。
                    k: 返すドキュメントの数。デフォルトは4。

                戻り値:
                    埋め込みに最も類似したドキュメントのリスト。
        """
        docs_and_scores = self.similarity_search_with_score_by_vector(embedding, k)
        return [doc for doc, _ in docs_and_scores]

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """クエリに最も類似したドキュメントを返す。

                引数:
                    query: 類似したドキュメントを検索するためのテキスト。
                    k: 返すドキュメントの数。デフォルトは4。

                戻り値:
                    クエリに最も類似したドキュメントのリスト。
        """
        docs_and_scores = self.similarity_search_with_score(query, k)
        return [doc for doc, _ in docs_and_scores]

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """最大限のマージナル関連性を使用して選択されたドキュメントを返します。

                最大限のマージナル関連性は、クエリとの類似性と選択されたドキュメント間の多様性を最適化します。

                引数:
                    embedding: 類似するドキュメントを検索するための埋め込み。
                    k: 返すドキュメントの数。デフォルトは4。
                    fetch_k: MMRアルゴリズムに渡すために取得するドキュメントの数。
                    lambda_mult: 0から1の間の数で、結果の多様性の程度を決定します。
                                0は最大の多様性、1は最小の多様性に対応します。
                                デフォルトは0.5。
                戻り値:
                    最大限のマージナル関連性によって選択されたドキュメントのリスト。
        """
        _, indices = self.index.search(np.array([embedding], dtype=np.float32), fetch_k)
        # -1 happens when not enough docs are returned.
        embeddings = [self.index.reconstruct(int(i)) for i in indices[0] if i != -1]
        mmr_selected = maximal_marginal_relevance(
            np.array([embedding], dtype=np.float32),
            embeddings,
            k=k,
            lambda_mult=lambda_mult,
        )
        selected_indices = [indices[0][i] for i in mmr_selected]
        docs = []
        for i in selected_indices:
            if i == -1:
                # This happens when not enough docs are returned.
                continue
            _id = self.index_to_docstore_id[i]
            doc = self.docstore.search(_id)
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            docs.append(doc)
        return docs

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """最大限のマージナル関連性を使用して選択されたドキュメントを返します。

                最大限のマージナル関連性は、クエリとの類似性と選択されたドキュメント間の多様性を最適化します。

                引数:
                    query: 類似するドキュメントを検索するためのテキスト。
                    k: 返すドキュメントの数。デフォルトは4。
                    fetch_k: MMRアルゴリズムに渡すために取得するドキュメントの数。
                    lambda_mult: 0から1の間の数で、結果の多様性の程度を決定します。
                                0は最大の多様性、1は最小の多様性に対応します。
                                デフォルトは0.5。
                戻り値:
                    最大限のマージナル関連性によって選択されたドキュメントのリスト。
        """
        embedding = self.embedding_function(query)
        docs = self.max_marginal_relevance_search_by_vector(
            embedding, k, fetch_k, lambda_mult=lambda_mult
        )
        return docs

    def merge_from(self, target: FAISS) -> None:
        """現在のFAISSオブジェクトに別のFAISSオブジェクトをマージします。

                対象のFAISSを現在のものに追加します。

                Args:
                    target: 現在のものにマージしたいFAISSオブジェクト

                Returns:
                    なし。
        """
        if not isinstance(self.docstore, AddableMixin):
            raise ValueError("Cannot merge with this type of docstore")
        # Numerical index for target docs are incremental on existing ones
        starting_len = len(self.index_to_docstore_id)

        # Merge two IndexFlatL2
        self.index.merge_from(target.index)

        # Create new id for docs from target FAISS object
        full_info = []
        for i in target.index_to_docstore_id:
            doc = target.docstore.search(target.index_to_docstore_id[i])
            if not isinstance(doc, Document):
                raise ValueError("Document should be returned")
            full_info.append((starting_len + i, str(uuid.uuid4()), doc))

        # Add information to docstore and index_to_docstore_id.
        self.docstore.add({_id: doc for _, _id, doc in full_info})
        index_to_id = {index: _id for index, _id, _ in full_info}
        self.index_to_docstore_id.update(index_to_id)

    @classmethod
    def __from(
        cls,
        texts: List[str],
        embeddings: List[List[float]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> FAISS:
        faiss = dependable_faiss_import()
        index = faiss.IndexFlatL2(len(embeddings[0]))
        index.add(np.array(embeddings, dtype=np.float32))
        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            documents.append(Document(page_content=text, metadata=metadata))
        index_to_id = {i: str(uuid.uuid4()) for i in range(len(documents))}
        docstore = InMemoryDocstore(
            {index_to_id[i]: doc for i, doc in enumerate(documents)}
        )
        return cls(embedding.embed_query, index, docstore, index_to_id, **kwargs)

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> FAISS:
        """生のドキュメントからFAISSラッパーを構築します。

                これは、ユーザーフレンドリーなインターフェースで、次のことができます。
                    1. ドキュメントを埋め込む。
                    2. メモリ内のdocstoreを作成する
                    3. FAISSデータベースを初期化する

                これは、すぐに始めるための簡単な方法です。

                例：
                    .. code-block:: python

                        from langchain import FAISS
                        from langchain.embeddings import OpenAIEmbeddings
                        embeddings = OpenAIEmbeddings()
                        faiss = FAISS.from_texts(texts, embeddings)
        """
        embeddings = embedding.embed_documents(texts)
        return cls.__from(
            texts,
            embeddings,
            embedding,
            metadatas,
            **kwargs,
        )

    @classmethod
    def from_embeddings(
        cls,
        text_embeddings: List[Tuple[str, List[float]]],
        embedding: Embeddings,
        metadata: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> FAISS:
        """生のドキュメントからFAISSラッパーを構築します。

                これは、次のようなユーザーフレンドリーなインターフェースです。
                    1. ドキュメントを埋め込む。
                    2. インメモリーのdocstoreを作成する
                    3. FAISSデータベースを初期化する

                これは、手っ取り早く始められる事を意図しています。

                例：
                    .. code-block:: python

                        from langchain import FAISS
                        from langchain.embeddings import OpenAIEmbeddings
                        embeddings = OpenAIEmbeddings()
                        text_embeddings = embeddings.embed_documents(texts)
                        text_embedding_pairs = list(zip(texts, text_embeddings))
                        faiss = FAISS.from_embeddings(text_embedding_pairs, embeddings)
        """
        texts = [t[0] for t in text_embeddings]
        embeddings = [t[1] for t in text_embeddings]
        return cls.__from(
            texts,
            embeddings,
            embedding,
            metadata,
            **kwargs,
        )

    def save_local(self, folder_path: str, index_name: str = "index") -> None:
        """FAISSインデックス、docstore、index_to_docstore_idをディスクに保存する。

                引数:
                    folder_path: インデックス、docstore、
                                index_to_docstore_idを保存するフォルダのパス。
                    index_name: インデックスファイル名を指定して保存する場合
        """
        path = Path(folder_path)
        path.mkdir(exist_ok=True, parents=True)

        # save index separately since it is not picklable
        faiss = dependable_faiss_import()
        faiss.write_index(
            self.index, str(path / "{index_name}.faiss".format(index_name=index_name))
        )

        # docstore と index_to_docstore_id を保存する
        with open(path / "{index_name}.pkl".format(index_name=index_name), "wb") as f:
            pickle.dump((self.docstore, self.index_to_docstore_id), f)

    @classmethod
    def load_local(
        cls, folder_path: str, embeddings: Embeddings, index_name: str = "index"
    ) -> FAISS:
        """FAISSインデックス、docstore、index_to_docstore_idをディスクに読み込む。

                引数:
                    folder_path: インデックス、docstore、index_to_docstore_id
                                　の読み込み元となるフォルダーパス
                    embeddings: クエリ生成時に使用する埋め込み
                    index_name: 特定のインデックスファイル名で保存する場合
        """
        path = Path(folder_path)
        # load index separately since it is not picklable
        faiss = dependable_faiss_import()
        index = faiss.read_index(
            str(path / "{index_name}.faiss".format(index_name=index_name))
        )

        # docstoreとindex_to_docstore_idを読み込む
        with open(path / "{index_name}.pkl".format(index_name=index_name), "rb") as f:
            docstore, index_to_docstore_id = pickle.load(f)
        return cls(embeddings.embed_query, index, docstore, index_to_docstore_id)

    def _similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Return docs and their similarity scores on a scale from 0 to 1.
        docsとその類似度スコアを0から1のスケールで返します。
        """
        if self.relevance_score_fn is None:
            raise ValueError(
                "normalize_score_fn must be provided to"
                " FAISS constructor to normalize scores"
            )
        docs_and_scores = self.similarity_search_with_score(query, k=k)
        return [(doc, self.relevance_score_fn(score)) for doc, score in docs_and_scores]
