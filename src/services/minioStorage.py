import os
from functools import cached_property
from typing import Any

from agent_proto import async_io, robust
from agent_proto.proxy import LazyProxy
from agent_proto.utils import setup_logging
from boto3 import client
from fastapi import UploadFile

from .redisCache import cache

logger = setup_logging(__name__)


def _det_content_type(filename: str) -> str:
    ext = filename.split(".")[-1]
    if ext in ("jpg", "jpeg", "png", "gif", "webp", "svg", "bmp", "ico"):
        return f"image/{ext}"
    if ext in ("mp4", "webm", "flv", "avi", "mov", "wmv", "mpg", "mpeg"):
        return f"video/{ext}"
    if ext in ("mp3", "wav", "flac", "aac", "ogg", "wma", "m4a"):
        return f"audio/{ext}"
    if ext in ("pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx"):
        return f"application/{ext}"
    if ext in ("zip", "tar", "gz", "rar", "7z"):
        return f"application/{ext}"
    if ext in (
        "txt",
        "csv",
        "json",
        "xml",
        "yaml",
        "yml",
        "html",
        "htm",
        "css",
        "js",
        "ts",
        "jsx",
        "tsx",
        "md",
        "rst",
    ):
        return f"text/{ext}"
    return "text/x-python" if ext == "py" else "application/octet-stream"


class ObjectStorage(LazyProxy[Any]):
    def __load__(self):
        return client(
            service_name="s3",
            endpoint_url="https://storage.oscarbahamonde.com",
            aws_access_key_id=os.environ.get("MINIO_ROOT_USER"),
            aws_secret_access_key=os.environ.get("MINIO_ROOT_PASSWORD"),
            region_name="us-east-1",
        )

    @cached_property
    def minio(self):
        return self.__load__()

    @async_io
    def put_object(
        self, *, key: str, data: bytes, content_type: str, bucket: str = "tera"
    ):
        self.minio.put_object(
            Bucket=bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
            ACL="public-read",
        )

    @async_io
    def generate_presigned_url(
        self, *, key: str, bucket: str = "tera", ttl: int = 3600
    ):
        return self.minio.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=ttl,
        )

    @robust
    async def put(self, *, key: str, file: UploadFile, bucket: str = "tera"):
        # sourcery skip: remove-redundant-if
        if not file.content_type and file.filename:
            content_type = _det_content_type(file.filename)
        elif not file.filename and file.content_type:
            content_type = file.content_type
        elif file.filename and file.content_type:
            content_type = file.content_type
        else:
            content_type = "application/octet-stream"
        data = await file.read()
        await self.put_object(
            key=key, data=data, content_type=content_type, bucket=bucket
        )
        return await self.generate_presigned_url(key=key, bucket=bucket)

    @robust
    @cache
    async def get(self, *, key: str, bucket: str = "tera"):
        return await self.generate_presigned_url(key=key, bucket=bucket)

    @async_io
    def remove_object(self, *, key: str, bucket: str = "tera"):
        self.minio.delete_object(Bucket=bucket, Key=key)

    @robust
    async def remove(self, *, key: str, bucket: str = "tera"):
        await self.remove_object(key=key, bucket=bucket)
        return True

    @async_io
    def list_objects(self, *, key: str, bucket: str = "tera"):
        return self.minio.list_objects(Bucket=bucket, Prefix=key)

    @robust
    async def list(self, *, key: str, bucket: str = "tera"):
        return await self.list_objects(key=key, bucket=bucket)
