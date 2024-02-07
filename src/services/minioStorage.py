import os
from functools import cached_property
from typing import Any

from agent_proto import async_io, robust
from agent_proto.proxy import LazyProxy
from agent_proto.utils import setup_logging
from boto3 import client

from .redisCache import cache

logger = setup_logging(__name__)


class ObjectStorage(LazyProxy[Any]):
    def __load__(self):
        return client(
            service_name="s3",
            endpoint_url="http://bucket:9000",
            aws_access_key_id=os.environ.get("MINIO_ROOT_USER"),
            aws_secret_access_key=os.environ.get("MINIO_ROOT_PASSWORD"),
            region_name="us-east-1",
        )

    @cached_property
    def minio(self):
        return self.__load__()

    @async_io
    def put_object(
        self, key: str, data: bytes, content_type: str, bucket: str = "tera"
    ):
        self.minio.put_object(
            Bucket=bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
            ACL="public-read",
            ContentDisposition="inline",
        )

    @async_io
    def generate_presigned_url(self, key: str, bucket: str = "tera", ttl: int = 3600):
        return self.minio.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=ttl,
        )

    @robust
    async def put(self, key: str, data: bytes, bucket: str = "tera"):
        await self.put_object(key, data, bucket)
        return await self.get(key, bucket)

    @robust
    @cache
    async def get(self, key: str, bucket: str = "tera"):
        return await self.generate_presigned_url(key, bucket)

    @async_io
    def remove_object(self, key: str, bucket: str = "tera"):
        self.minio.delete_object(Bucket=bucket, Key=key)

    @robust
    async def remove(self, key: str, bucket: str = "tera"):
        await self.remove_object(key, bucket)
        return True

    @async_io
    def list_objects(self, key: str, bucket: str = "tera"):
        return self.minio.list_objects(Bucket=bucket, Prefix=key)

    @robust
    async def list(self, key: str, bucket: str = "tera"):
        return await self.list_objects(key, bucket)
