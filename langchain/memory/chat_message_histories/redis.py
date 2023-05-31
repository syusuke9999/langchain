import json
import logging
import os
from typing import List, Optional

from langchain.schema import (
    AIMessage,
    BaseChatMessageHistory,
    BaseMessage,
    HumanMessage,
    _message_to_dict,
    messages_from_dict,
)

logger = logging.getLogger(__name__)


class RedisChatMessageHistory(BaseChatMessageHistory):
    def __init__(
            self,
            session_id: str,
            key_prefix: str = "message_store:",
            ttl: Optional[int] = None,
            redis_server_url: Optional[str] = None,
            redis_server_password: Optional[str] = None,
            redis_port_number: Optional[int] = 0,
    ):
        try:
            import redis
        except ImportError:
            raise ImportError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            )

        # Get password, host, and port from environment variables or use the provided ones
        password = os.getenv('REDIS_PASSWORD', redis_server_password)
        host = os.getenv('REDIS_HOST', redis_server_url)
        port = os.getenv('REDIS_PORT', redis_port_number)

        # Construct the url
        url = f"redis://:{password}@{host}:{port}"

        try:
            self.redis_client = redis.Redis.from_url(url=url)
        except redis.exceptions.ConnectionError as error:
            logger.error(error)

        self.session_id = session_id
        self.key_prefix = key_prefix
        self.ttl = ttl

    @property
    def key(self) -> str:
        """Construct the record key to use"""
        return self.key_prefix + self.session_id

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve the messages from Redis"""
        _items = self.redis_client.lrange(self.key, 0, -1)
        items = [json.loads(m.decode("utf-8")) for m in _items[::-1]]
        messages = messages_from_dict(items)
        return messages

    def add_user_message(self, message: str) -> None:
        self.append(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        self.append(AIMessage(content=message))

    def append(self, message: BaseMessage) -> None:
        """Append the message to the record in Redis"""
        self.redis_client.lpush(self.key, json.dumps(_message_to_dict(message)))
        if self.ttl:
            self.redis_client.expire(self.key, self.ttl)

    def clear(self) -> None:
        """Clear session memory from Redis"""
        self.redis_client.delete(self.key)
