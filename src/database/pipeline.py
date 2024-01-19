from typing import Generic, TypeVar, Callable

import numpy as np
import tiktoken
import torch
import transformers  # type: ignore
from pydantic import BaseModel, ConfigDict
from sentence_transformers import SentenceTransformer  # type: ignore

NN = TypeVar("NN", bound=torch.nn.Module)
T = TypeVar("T")


class Launch(Generic[NN], BaseModel):
    tokenizer: tiktoken.Encoding
    attention_mechanism: NN
    embedding: SentenceTransformer
    instance: transformers.AutoModelForCausalLM
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    dropout: torch.nn.Dropout,
    func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = torch.matmul,
) -> torch.Tensor:
    scores = func(q, k.transpose(-2, -1)) / np.sqrt(q.size(-1))
    scores = scores.masked_fill(mask == 0, -1e9)
    scores = dropout(torch.softmax(scores, dim=-1))
    return func(scores, v)
