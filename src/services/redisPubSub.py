"""PubSub engine"""

from typing import AsyncGenerator

import aioredis
from agent_proto.proxy import LazyProxy
from agent_proto.utils import handle_errors, process_time, setup_logging
from aioredis.client import PubSub

logger = setup_logging(__name__)

pool = aioredis.Redis.from_url("redis://redis:6379/0")  # type: ignore


class RedisPubSub(LazyProxy[PubSub]):
    """
    PubSub channel to send function call results to the client.
    """

    pubsub: PubSub
    namespace: str

    def __init__(self, namespace: str):
        self.namespace = namespace
        self.pubsub = self.__load__()
        super().__init__()

    def __load__(self):
        """
        Lazy loading of the PubSub object.
        """
        return pool.pubsub()  # type: ignore

    async def sub(self) -> AsyncGenerator[str, None]:
        """
        Subscribes to the PubSub channel and yields messages as they come in.
        """
        await self.pubsub.subscribe(self.namespace)  # type: ignore
        logger.info("Subscribed to %s", self.namespace)
        async for message in self.pubsub.listen():  # type: ignore
            try:
                data = message.get("data")  # type: ignore
                yield data.decode("utf-8") if isinstance(data, bytes) else data
            except (KeyError, AssertionError, UnicodeDecodeError, AttributeError):
                continue

    @handle_errors
    async def _send(self, message: str) -> None:
        """
        Protected method to send a message to the PubSub channel.
        """
        await pool.publish(self.namespace, message)  # type: ignore
        logger.info("Message published to %s", self.namespace)

    @process_time
    @handle_errors
    async def pub(self, message: str) -> None:
        """
        Public method to send a function call result to the PubSub channel.
        """
        logger.info("Sending message to %s", self.namespace)
        await self._send(message)
        await self.pubsub.unsubscribe(self.namespace)  # type: ignore
