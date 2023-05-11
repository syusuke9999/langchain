import csv
from typing import Dict, List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class CSVLoader(BaseLoader):
    """CSVファイルをドキュメントのリストに読み込みます。

        各ドキュメントはCSVファイルの1行を表します。各行はキー/値のペアに変換され、
        ドキュメントのpage_contentの新しい行に出力されます。

        デフォルトでは、csvから読み込まれた各ドキュメントのソースは、
        `file_path`引数の値に設定されます。
        これを上書きするには、`source_column`引数をCSVファイルの列名に設定します。
        その後、各ドキュメントのソースは、`source_column`で指定された名前の列の値に設定されます。

        出力例:
            .. code-block:: txt

                column1: value1
                column2: value2
                column3: value3
        """

    def __init__(
        self,
        file_path: str,
        source_column: Optional[str] = None,
        csv_args: Optional[Dict] = None,
        encoding: Optional[str] = None,
    ):
        self.file_path = file_path
        self.source_column = source_column
        self.encoding = encoding
        self.csv_args = csv_args or {}

    def load(self) -> List[Document]:
        """ドキュメントオブジェクトへデータを読み込む"""

        docs = []
        with open(self.file_path, newline="", encoding=self.encoding) as csvfile:
            csv_reader = csv.DictReader(csvfile, **self.csv_args)  # type: ignore
            for i, row in enumerate(csv_reader):
                content = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in row.items())
                try:
                    source = (
                        row[self.source_column]
                        if self.source_column is not None
                        else self.file_path
                    )
                except KeyError:
                    raise ValueError(
                        f"Source column '{self.source_column}' not found in CSV file."
                    )
                metadata = {"source": source, "row": i}
                doc = Document(page_content=content, metadata=metadata)
                docs.append(doc)

        return docs
