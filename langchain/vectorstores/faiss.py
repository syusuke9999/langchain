"""Wrapper around FAISS vector database."""
from __future__ import annotations

import math
import os
import pickle
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import time
from tqdm import tqdm

import numpy as np

from langchain.docstore.base import AddableMixin, Docstore
from langchain.docstore.document import Document
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.utils import maximal_marginal_relevance


def dependable_faiss_import(no_avx2: Optional[bool] = None) -> Any:
    """
        faissが利用可能な場合はインポートし、それ以外の場合はエラーを発生させます。
        FAISS_NO_AVX2環境変数が設定されている場合、AVX2の最適化なしでFAISSをロードすると考えられます。

        Args:
            no_avx2: AVX2の最適化なしでFAISSを厳密にロードするため、
                vectorstoreはポータブルで他のデバイスと互換性があります。
    """
    if no_avx2 is None and "FAISS_NO_AVX2" in os.environ:
        no_avx2 = bool(os.getenv("FAISS_NO_AVX2"))

    try:
        if no_avx2:
            from faiss import swigfaiss as faiss
        else:
            import faiss
    except ImportError:
        raise ImportError(
            "Could not import faiss python package. "
            "Please install it with `pip install faiss` "
            "or `pip install faiss-cpu` (depending on Python version)."
        )
    return faiss


def _default_relevance_score_fn(score: float) -> float:
    """[0, 1]のスケールで類似度スコアを返します。"""
    # '正しい'関連性関数は、いくつかの要素によって異なる場合があります。
    # - VectorStoreで使用される距離/類似度メトリック
    # - 埋め込みのスケール（OpenAIの場合は単位ノルム化されていますが、他の多くの場合はそうではありません）
    # - 埋め込みの次元数
    # - その他
    # この関数は、正規化された埋め込みのユークリッドノルム（0が最も類似しており、sqrt(2)が最も類似していない）を
    # 類似度関数（0から1）に変換します。
    return 1.0 - score / math.sqrt(2)


class FAISS(VectorStore):
    """FAISSベクトルデータベースのラッパーです。
        使用するには、``faiss``のPythonパッケージをインストールする必要があります。
        例：
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
        normalize_L2: bool = False,
    ):
        """必要なコンポーネントで初期化します。"""
        self.embedding_function = embedding_function
        self.index = index
        self.docstore = docstore
        self.index_to_docstore_id = index_to_docstore_id
        self.relevance_score_fn = relevance_score_fn
        self._normalize_L2 = normalize_L2

    def __add(
        self,
        texts: Iterable[str],
        embeddings: Iterable[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
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
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        # Add to the index, the index_to_id mapping, and the docstore.
        starting_len = len(self.index_to_docstore_id)
        faiss = dependable_faiss_import()
        vector = np.array(embeddings, dtype=np.float32)
        if self._normalize_L2:
            faiss.normalize_L2(vector)
        self.index.add(vector)
        # インデックス、ID、およびドキュメントのリストを取得
        full_info = [(starting_len + i, ids[i], doc) for i, doc in enumerate(documents)]
        # docstoreとインデックスに情報を追加します
        self.docstore.add({_id: doc for _, _id, doc in full_info})
        index_to_id = {index: _id for index, _id, _ in full_info}
        self.index_to_docstore_id.update(index_to_id)
        return [_id for _, _id, _ in full_info]

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """埋め込みを介してさらにテキストを実行し、ベクトルストアに追加します。
                Args:
                    texts: ベクトルストアに追加するための文字列のイテラブル。
                    metadatas: テキストに関連するメタデータのオプションのリスト。
                    ids: ユニークなIDのオプションのリスト。
                Returns:
                    テキストをベクトルストアに追加した結果のIDのリスト。
        """
        if not isinstance(self.docstore, AddableMixin):
            raise ValueError(
                "If trying to add texts, the underlying docstore should support "
                f"adding items, which {self.docstore} does not"
            )
        # 埋め込みを作成し、ドキュメントを生成します。
        embeddings = [self.embedding_function(text) for text in texts]
        return self.__add(texts, embeddings, metadatas=metadatas, ids=ids, **kwargs)

    def add_embeddings(
        self,
        text_embeddings: Iterable[Tuple[str, List[float]]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """埋め込みを介してさらにテキストを実行し、ベクトルストアに追加します。
                Args:
                    text_embeddings: ベクトルストアに追加するための文字列と埋め込みのペアのイテラブル。
                    metadatas: テキストに関連するメタデータのオプションのリスト。
                    ids: ユニークなIDのオプションのリスト。
                Returns:
                    テキストをベクトルストアに追加した結果のIDのリスト。
        """
        if not isinstance(self.docstore, AddableMixin):
            raise ValueError(
                "If trying to add texts, the underlying docstore should support "
                f"adding items, which {self.docstore} does not"
            )
        # 埋め込みを作成し、ドキュメントを生成します。
        texts, embeddings = zip(*text_embeddings)

        return self.__add(texts, embeddings, metadatas=metadatas, ids=ids, **kwargs)

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """クエリに最も類似したドキュメントを返します。
                Args:
                    embedding: 類似したドキュメントを検索するための埋め込みベクトル。
                    k: 返すドキュメントの数。デフォルトは4です。
                    filter (Optional[Dict[str, Any]]): メタデータでフィルタリングします。デフォルトはNoneです。
                    fetch_k: (Optional[int]) フィルタリングする前に取得するドキュメントの数。デフォルトは20です。
                    **kwargs: 類似度検索に渡されるkwargs。以下を含むことができます：
                        score_threshold: オプション、0から1までの浮動小数点値で、
                            検索結果のドキュメントセットをフィルタリングするための閾値
                Returns:
                    クエリに最も類似したドキュメントのリスト。各ドキュメントのL2距離が浮動小数点数で表示されます。スコアが低いほど類似度が高いことを示します。
        """
        faiss = dependable_faiss_import()
        vector = np.array([embedding], dtype=np.float32)
        if self._normalize_L2:
            faiss.normalize_L2(vector)
        scores, indices = self.index.search(vector, k if filter is None else fetch_k)
        docs = []
        for j, i in enumerate(indices[0]):
            if i == -1:
                # これは、十分なドキュメントが返されないときに発生します。
                continue
            _id = self.index_to_docstore_id[i]
            doc = self.docstore.search(_id)
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            if filter is not None:
                filter = {
                    key: [value] if not isinstance(value, list) else value
                    for key, value in filter.items()
                }
                if all(doc.metadata.get(key) in value for key, value in filter.items()):
                    docs.append((doc, scores[0][j]))
            else:
                docs.append((doc, scores[0][j]))

        score_threshold = kwargs.get("score_threshold")
        if score_threshold is not None:
            docs = [
                (doc, similarity)
                for doc, similarity in docs
                if similarity >= score_threshold
            ]
        return docs[:k]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """クエリに最も類似したドキュメントを返します。
                Args:
                    query: 類似したドキュメントを検索するテキスト。
                    k: 返すドキュメントの数。デフォルトは4です。
                    filter (Optional[Dict[str, str]]): メタデータでフィルタリングします。デフォルトはNoneです。
                    fetch_k: (Optional[int]) フィルタリングする前に取得するドキュメントの数。デフォルトは20です。
                Returns:
                    クエリに最も類似したドキュメントのリスト。L2距離が浮動小数点数で表示されます。スコアが低いほど類似度が高いことを示します。
        """
        embedding = self.embedding_function(query)
        docs = self.similarity_search_with_score_by_vector(
            embedding,
            k,
            filter=filter,
            fetch_k=fetch_k,
            **kwargs,
        )
        return docs

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Document]:
        """埋め込みベクトルに最も類似したドキュメントを返します。
                Args:
                    embedding: 類似したドキュメントを検索するための埋め込み。
                    k: 返すドキュメントの数。デフォルトは4です。
                    filter (Optional[Dict[str, str]]): メタデータでフィルタリングします。デフォルトはNoneです。
                    fetch_k: (Optional[int]) フィルタリングする前に取得するドキュメントの数。デフォルトは20です。
                Returns:
                    埋め込みに最も類似したドキュメントのリスト。
        """
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding,
            k,
            filter=filter,
            fetch_k=fetch_k,
            **kwargs,
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Document]:
        """クエリに最も類似したドキュメントを返します。
                Args:
                    query: 類似したドキュメントを検索するテキスト。
                    k: 返すドキュメントの数。デフォルトは4です。
                    filter: (Optional[Dict[str, str]]): メタデータでフィルタリングします。デフォルトはNoneです。
                    fetch_k: (Optional[int]) フィルタリングする前に取得するドキュメントの数。デフォルトは20です。
                Returns:
                    クエリに最も類似したドキュメントのリスト。
        """
        docs_and_scores = self.similarity_search_with_score(
            query, k, filter=filter, fetch_k=fetch_k, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    def max_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: List[float],
        *,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """最大限のマージナル関連性を使用して選択されたドキュメントとその類似度スコアを返します。
                最大限のマージナル関連性は、クエリに対する類似度と選択されたドキュメントの多様性を最適化します。
                Args:
                    embedding: 類似したドキュメントを検索するための埋め込み。
                    k: 返すドキュメントの数。デフォルトは4です。
                    fetch_k: フィルタリングする前に取得するドキュメントの数を指定するパラメータ。
                             MMRアルゴリズムに渡されます。
                    lambda_mult: 0から1までの数値で、結果の多様性の度合いを決定するパラメータです。
                                0は最大の多様性を示し、1は最小の多様性を示します。デフォルトは0.5です。
                Returns:
                    最大限のマージナル関連性によって選択されたドキュメントとその類似度スコアのリスト。
        """
        scores, indices = self.index.search(
            np.array([embedding], dtype=np.float32),
            fetch_k if filter is None else fetch_k * 2,
        )
        if filter is not None:
            filtered_indices = []
            for i in indices[0]:
                if i == -1:
                    # これは、十分なドキュメントが返されないときに発生します。
                    continue
                _id = self.index_to_docstore_id[i]
                doc = self.docstore.search(_id)
                if not isinstance(doc, Document):
                    raise ValueError(f"Could not find document for id {_id}, got {doc}")
                if all(doc.metadata.get(key) == value for key, value in filter.items()):
                    filtered_indices.append(i)
            indices = np.array([filtered_indices])
        # -1は、十分なドキュメントが返されないときに発生します。
        embeddings = [self.index.reconstruct(int(i)) for i in indices[0] if i != -1]
        mmr_selected = maximal_marginal_relevance(
            np.array([embedding], dtype=np.float32),
            embeddings,
            k=k,
            lambda_mult=lambda_mult,
        )
        selected_indices = [indices[0][i] for i in mmr_selected]
        selected_scores = [scores[0][i] for i in mmr_selected]
        docs_and_scores = []
        for i, score in zip(selected_indices, selected_scores):
            if i == -1:
                # これは、十分なドキュメントが返されないときに発生します。
                continue
            _id = self.index_to_docstore_id[i]
            doc = self.docstore.search(_id)
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            docs_and_scores.append((doc, score))
        return docs_and_scores

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        最大限のマージナル関連性を使用して選択されたドキュメントを返します。
                最大限のマージナル関連性は、クエリに対する類似度と選択されたドキュメントの多様性を最適化します。
                Args:
                    embedding: 類似したドキュメントを検索するための埋め込み。
                    k: 返すドキュメントの数。デフォルトは4です。
                    fetch_k: フィルタリングする前に取得するドキュメントの数を指定するパラメータ。
                             MMRアルゴリズムに渡されます。
                    lambda_mult: 0から1までの数値で、結果の多様性の度合いを決定するパラメータです。
                                0は最大の多様性を示し、1は最小の多様性を示します。デフォルトは0.5です。
                Returns:
                    最大限のマージナル関連性によって選択されたドキュメントのリスト。
        """
        docs_and_scores = self.max_marginal_relevance_search_with_score_by_vector(
            embedding, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter
        )
        return [doc for doc, _ in docs_and_scores]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        最大限のマージナル関連性を使用して選択されたドキュメントを返します。
                最大限のマージナル関連性は、クエリに対する類似度と選択されたドキュメントの多様性を最適化します。
                Args:
                    query: 類似したドキュメントを検索するテキスト。
                    k: 返すドキュメントの数。デフォルトは4です。
                    fetch_k: フィルタリング（必要な場合）のために取得するドキュメントの数を指定するパラメータ。
                             MMRアルゴリズムに渡されます。
                    lambda_mult: 0から1までの数値で、結果の多様性の度合いを決定するパラメータです。
                                0は最大の多様性を示し、1は最小の多様性を示します。デフォルトは0.5です。
                Returns:
                    最大限のマージナル関連性によって選択されたドキュメントのリスト。
        """
        embedding = self.embedding_function(query)
        docs = self.max_marginal_relevance_search_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )
        return docs

    def merge_from(self, target: FAISS) -> None:
        """
        別のFAISSオブジェクトを現在のオブジェクトにマージします。
                ターゲットのFAISSを現在のFAISSに追加します。
                Args:
                    target: 現在のオブジェクトにマージしたいFAISSオブジェクト
                Returns:
                    None.
        """
        if not isinstance(self.docstore, AddableMixin):
            raise ValueError("Cannot merge with this type of docstore")
        # ターゲットドキュメントの数値インデックスは、既存のものに対して増加します。
        starting_len = len(self.index_to_docstore_id)

        # 2つのIndexFlatL2をマージします
        self.index.merge_from(target.index)

        # ターゲットのFAISSオブジェクトからIDとドキュメントを取得します。
        full_info = []
        for i, target_id in target.index_to_docstore_id.items():
            doc = target.docstore.search(target_id)
            if not isinstance(doc, Document):
                raise ValueError("Document should be returned")
            full_info.append((starting_len + i, target_id, doc))

        # docstoreとindex_to_docstore_idに情報を追加します
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
        ids: Optional[List[str]] = None,
        normalize_L2: bool = False,
        **kwargs: Any,
    ) -> FAISS:
        faiss = dependable_faiss_import()
        index = faiss.IndexFlatL2(len(embeddings[0]))
        vector = np.array(embeddings, dtype=np.float32)
        if normalize_L2:
            faiss.normalize_L2(vector)
        index.add(vector)
        documents = []
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            documents.append(Document(page_content=text, metadata=metadata))
        index_to_id = dict(enumerate(ids))
        docstore = InMemoryDocstore(dict(zip(index_to_id.values(), documents)))
        return cls(
            embedding.embed_query,
            index,
            docstore,
            index_to_id,
            normalize_L2=normalize_L2,
            **kwargs,
        )

    @classmethod
    def from_texts(
            cls,
            texts: List[str],
            embedding: Embeddings,
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            batch_size: int = 10,
            wait_time: int = 1,
            **kwargs: Any,
    ) -> 'FAISS':
        """
        生のドキュメントからFAISSラッパーを構築します。
        これはユーザーフレンドリーなインターフェースで、次の操作を行います:
            1. ドキュメントを埋め込みます。
            2. メモリ内のdocstoreを作成します。
            3. FAISSデータベースを初期化します。
        これは、すぐに始めるための方法を提供することを目的としています。
        例:
            .. code-block:: python
                from langchain import FAISS
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                faiss = FAISS.from_texts(texts, embeddings)
        """
        all_embeddings = []

        # バッチごとにテキストデータをFAISSインデックスに追加します。
        for i in tqdm(range(0, len(texts), batch_size)):
            # バッチを作成します。
            batch_texts = texts[i:i + batch_size]
            # バッチのテキストデータを埋め込みに変換します。
            batch_embeddings = embedding.embed_documents(batch_texts)
            # 埋め込みを全体のリストに追加します。
            all_embeddings.extend(batch_embeddings)
            # 次のバッチの処理まで待ちます。
            time.sleep(wait_time)

        # 全てのテキストデータの埋め込みからFAISSインデックスを作成します。
        return cls.__from(
            texts,
            all_embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            **kwargs
        )

    @classmethod
    def from_embeddings(
        cls,
        text_embeddings: List[Tuple[str, List[float]]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> FAISS:
        """
        生のドキュメントからFAISSラッパーを構築します。
                これは次のようなユーザーフレンドリーなインターフェースです：
                    1. ドキュメントを埋め込む。
                    2. メモリ内のdocstoreを作成する。
                    3. FAISSデータベースを初期化する。
                これは、すぐに始めるための簡単な方法を提供することを目的としています。
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
            metadatas=metadatas,
            ids=ids,
            **kwargs,
        )

    def save_local(self, folder_path: str, index_name: str = "index") -> None:
        """
        FAISSインデックス、docstore、およびindex_to_docstore_idをディスクに保存します。
                Args:
                    folder_path: インデックス、docstore、およびindex_to_docstore_idを保存するためのフォルダーパス。
                    index_name: 特定のインデックスファイル名で保存するためのインデックス名
        """
        path = Path(folder_path)
        path.mkdir(exist_ok=True, parents=True)
        # インデックスはpicklableではないため、別々に保存します。
        faiss = dependable_faiss_import()
        faiss.write_index(
            self.index, str(path / "{index_name}.faiss".format(index_name=index_name))
        )
        # docstoreとindex_to_docstore_idを保存します。
        with open(path / "{index_name}.pkl".format(index_name=index_name), "wb") as f:
            pickle.dump((self.docstore, self.index_to_docstore_id), f)

    @classmethod
    def load_local(
        cls, folder_path: str, embeddings: Embeddings, index_name: str = "index"
    ) -> FAISS:
        """
        ディスクからFAISSインデックス、docstore、およびindex_to_docstore_idを読み込みます。
                Args:
                    folder_path: インデックス、docstore、およびindex_to_docstore_idを読み込むためのフォルダーパス。
                    embeddings: クエリ生成時に使用する埋め込み
                    index_name: 特定のインデックスファイル名で保存するためのインデックス名
        """
        path = Path(folder_path)
        # インデックスはpicklableではないため、別々に読み込みます。
        faiss = dependable_faiss_import()
        index = faiss.read_index(
            str(path / "{index_name}.faiss".format(index_name=index_name))
        )

        # docstoreとindex_to_docstore_idを読み込みます。
        with open(path / "{index_name}.pkl".format(index_name=index_name), "rb") as f:
            docstore, index_to_docstore_id = pickle.load(f)
        return cls(embeddings.embed_query, index, docstore, index_to_docstore_id)

    def _similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """0から1までのスケールでドキュメントとその類似度スコアを返します"""
        if self.relevance_score_fn is None:
            raise ValueError(
                "normalize_score_fn must be provided to"
                " FAISS constructor to normalize scores"
            )
        docs_and_scores = self.similarity_search_with_score(
            query,
            k=k,
            filter=filter,
            fetch_k=fetch_k,
            **kwargs,
        )
        return [(doc, self.relevance_score_fn(score)) for doc, score in docs_and_scores]
