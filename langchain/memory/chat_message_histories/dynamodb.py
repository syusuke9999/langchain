import logging
from typing import List

from langchain.schema import (
    AIMessage,
    BaseChatMessageHistory,
    BaseMessage,
    HumanMessage,
    _message_to_dict,
    messages_from_dict,
    messages_to_dict,
)

logger = logging.getLogger(__name__)


class DynamoDBChatMessageHistory(BaseChatMessageHistory):
    """AWS DynamoDBに履歴を保存するチャットメッセージ履歴。
        このクラスは、`table_name`という名前のDynamoDBテーブルと、
        `SessionId`というパーティションキーが存在することを想定しています。

        引数:
            table_name: DynamoDBテーブルの名前
            session_id: 1つのチャットセッションのメッセージを保存するために使用される任意のキー。
    """

    def __init__(self, table_name: str, session_id: str):
        import boto3

        client = boto3.resource("dynamodb")
        self.table = client.Table(table_name)
        self.session_id = session_id

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """DynamoDBからメッセージを取得する"""
        from botocore.exceptions import ClientError

        try:
            response = self.table.get_item(Key={"SessionId": self.session_id})
        except ClientError as error:
            if error.response["Error"]["Code"] == "ResourceNotFoundException":
                logger.warning("No record found with session id: %s", self.session_id)
            else:
                logger.error(error)

        if response and "Item" in response:
            items = response["Item"]["History"]
        else:
            items = []

        messages = messages_from_dict(items)
        return messages

    def add_user_message(self, message: str) -> None:
        self.append(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        self.append(AIMessage(content=message))

    def append(self, message: BaseMessage) -> None:
        """DynamoDBのレコードにメッセージを追加する"""
        from botocore.exceptions import ClientError

        messages = messages_to_dict(self.messages)
        _message = _message_to_dict(message)
        messages.append(_message)

        try:
            self.table.put_item(
                Item={"SessionId": self.session_id, "History": messages}
            )
        except ClientError as err:
            logger.error(err)

    def clear(self) -> None:
        """DynamoDBからセッションメモリを消去する"""
        from botocore.exceptions import ClientError

        try:
            self.table.delete_item(Key={"SessionId": self.session_id})
        except ClientError as err:
            logger.error(err)
