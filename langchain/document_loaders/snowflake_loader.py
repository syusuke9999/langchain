from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple

from snowflake.connector import DictCursor

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class SnowflakeLoader(BaseLoader):
    """Loads a query result from Snowflake into a list of documents.

    Each document represents one row of the result. The `page_content_columns`
    are written into the `page_content` of the document. The `metadata_columns`
    are written into the `metadata` of the document. By default, all columns
    are written into the `page_content` and none into the `metadata`.

    """

    def __init__(
        self,
        query: str,
        user: str,
        password: str,
        account: str,
        warehouse: str,
        role: str,
        database: str,
        schema: str,
        parameters: Optional[Dict[str, Any]] = None,
        page_content_columns: Optional[List[str]] = None,
        metadata_columns: Optional[List[str]] = None,
    ):
        """Initialize Snowflake document loader.

        Args:
            query: The query to run in Snowflake.
            user: Snowflake user.
            password: Snowflake password.
            account: Snowflake account.
            warehouse: Snowflake warehouse.
            role: Snowflake role.
            database: Snowflake database
            schema: Snowflake schema
            page_content_columns: Optional. Columns written to Document `page_content`.
            metadata_columns: Optional. Columns written to Document `metadata`.
        """
        self.query = query
        self.user = user
        self.password = password
        self.account = account
        self.warehouse = warehouse
        self.role = role
        self.database = database
        self.schema = schema
        self.parameters = parameters
        self.page_content_columns = page_content_columns
        self.metadata_columns = metadata_columns

    def _execute_query(self) -> List[Dict[str, Any]]:
        try:
            import snowflake.connector
        except ImportError as ex:
            raise ValueError(
                "Could not import snowflake-connector-python package. "
                "Please install it with `pip install snowflake-connector-python`."
            ) from ex

        conn = snowflake.connector.connect(
            user=self.user,
            password=self.password,
            account=self.account,
            warehouse=self.warehouse,
            role=self.role,
            database=self.database,
            schema=self.schema,
            parameters=self.parameters,
        )
        query_result = []
        try:
            cur = conn.cursor(DictCursor)
            cur.execute("USE DATABASE " + self.database)
            cur.execute("USE SCHEMA " + self.schema)
            cur.execute(self.query, self.parameters)
            query_result = cur.fetchall()
            query_result = [
                {k.lower(): v for k, v in item.items()} for item in query_result
            ]
        except Exception as e:
            print(f"An error occurred: {e}")
            query_result = []
        finally:
            cur.close()
        return query_result

    def _get_columns(
        self, query_result: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str]]:
        page_content_columns = (
            self.page_content_columns if self.page_content_columns else []
        )
        metadata_columns = self.metadata_columns if self.metadata_columns else []
        if page_content_columns is None and query_result:
            page_content_columns = list(query_result[0].keys())
        if metadata_columns is None:
            metadata_columns = []
        return page_content_columns or [], metadata_columns

    def lazy_load(self) -> Iterator[Document]:
        query_result = self._execute_query()
        if isinstance(query_result, Exception):
            print(f"An error occurred during the query: {query_result}")
            return []
        page_content_columns, metadata_columns = self._get_columns(query_result)
        for row in query_result:
            page_content = "\n".join(
                f"{k}: {v}" for k, v in row.items() if k in page_content_columns
            )
            metadata = {k: v for k, v in row.items() if k in metadata_columns}
            doc = Document(page_content=page_content, metadata=metadata)
            yield doc
