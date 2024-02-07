from __future__ import annotations

import base64
import os
import pathlib
import shutil
import subprocess
import tempfile
from typing import (
    Any,
    Callable,
    Dict,
    ForwardRef,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    get_type_hints,
    no_type_check,
    no_type_check_decorator,
    overload,
)

from pydantic import BaseModel, ConfigDict, Field, computed_field, create_model


class FileTree(BaseModel):
    path: str
    name: str
    is_dir: bool
    content: str | List[FileTree] = Field(...)

    @classmethod
    def from_path(cls, path: str, depth: Optional[int] = None) -> FileTree:
        if depth is None:
            depth = path.count("/")
        if os.path.isdir(path):
            content = [
                FileTree.from_path(os.path.join(path, child), depth=depth + 1)
                for child in os.listdir(path)
            ]
        else:
            try:
                with open(path, "r") as f:
                    content = f.read()
            except UnicodeDecodeError:
                content = base64.b64encode(open(path, "rb").read()).decode()
        return cls(
            path=path,
            name=os.path.basename(path),
            is_dir=os.path.isdir(path),
            content=content,
        )


TREE = FileTree.from_path("src")

print(TREE.model_dump_json(indent=2))
