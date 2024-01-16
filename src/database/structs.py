import json
from typing import Any

import numpy as np
import tiktoken
import torch
import transformers  # type: ignore
from pydantic import BaseModel, ConfigDict
from sentence_transformers import SentenceTransformer  # type: ignore


class LLMEncoder(json.JSONEncoder):
    @torch.no_grad()  # type: ignore
    def default(self, o: Any):  # type: ignore
        if isinstance(o, tiktoken.Encoding):
            return {
                "name": o.name,
                "max_token_value": o.max_token_value,
                "special_tokens_set": list(o.special_tokens_set),
                "n_vocab": o.n_vocab,
            }
        if isinstance(o, torch.Tensor):
            return o.tolist()  # type: ignore
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, transformers.PreTrainedModel):
            return o.config.to_dict()  # type: ignore
        if isinstance(o, SentenceTransformer):
            return {
                "max_seq_length": o.max_seq_length,
                "word_embedding_dimension": o.word_embedding_dimension,
                "dump_patches": o.dump_patches,
            }
        if isinstance(o, transformers.PreTrainedTokenizer):
            return {
                "added_token_decoder": o.added_tokens_decoder,
                "added_tokens_encoder": o.added_tokens_encoder,
                "all_special_ids": o.all_special_ids,
                "all_special_tokens": o.all_special_tokens,
                "bos_token": o.bos_token,
                "cls_token": o.cls_token,
                "eos_token": o.eos_token,
                "mask_token": o.mask_token,
                "max_len_single_sentence": o.max_len_single_sentence,
                "max_len_sentences_pair": o.max_len_sentences_pair,
                "pad_token": o.pad_token,
                "pad_token_id": o.pad_token_id,
                "pad_token_type_id": o.pad_token_type_id,
                "sep_token": o.sep_token,
                "unk_token": o.unk_token,
                "unk_token_id": o.unk_token_id,
                "vocab_files_names": o.vocab_files_names,
                "vocab_size": o.vocab_size,
            }
        if isinstance(o, transformers.PretrainedConfig):
            return o.to_dict()
        if isinstance(o, transformers.PreTrainedTokenizerFast):
            return {
                "added_tokens_decoder": o.added_tokens_decoder,
                "added_tokens_encoder": o.added_tokens_encoder,
                "all_special_ids": o.all_special_ids,
                "all_special_tokens": o.all_special_tokens,
                "bos_token": o.bos_token,
                "cls_token": o.cls_token,
                "eos_token": o.eos_token,
                "mask_token": o.mask_token,
                "max_len_single_sentence": o.max_len_single_sentence,
                "max_len_sentences_pair": o.max_len_sentences_pair,
                "pad_token": o.pad_token,
                "pad_token_id": o.pad_token_id,
                "pad_token_type_id": o.pad_token_type_id,
                "sep_token": o.sep_token,
                "unk_token": o.unk_token,
                "unk_token_id": o.unk_token_id,
                "vocab_files_names": o.vocab_files_names,
                "vocab_size": o.vocab_size,
            }
        if isinstance(o, torch.nn.Module):
            return o.state_dict()
        if isinstance(o, torch.optim.Optimizer):
            return o.state_dict()
        if isinstance(o, torch.Tensor):
            return o.tolist()  # type: ignore
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


class TypeDef(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
        json_encoders={
            np.ndarray: LLMEncoder().encode,
            torch.Tensor: LLMEncoder().encode,
            transformers.PreTrainedModel: LLMEncoder().encode,
            transformers.PreTrainedTokenizer: LLMEncoder().encode,
            transformers.PretrainedConfig: LLMEncoder().encode,
            transformers.PreTrainedTokenizerFast: LLMEncoder().encode,
            torch.nn.Module: LLMEncoder().encode,
            torch.optim.Optimizer: LLMEncoder().encode,
        },
    )


class LLMEncodingTypeDef(TypeDef):
    eot_token: int
    name: str
    max_token_value: int
    special_tokens_set: list[int]
    n_vocab: int


class LLMAttentionMechanismTypeDef(TypeDef):
    state: dict[str, torch.Tensor]


class LLMEmbeddingTypeDef(TypeDef):
    name: str
    max_seq_length: int
    word_embedding_dimension: int
    dump_patches: bool


class LLMInstanceTypeDef(TypeDef):
    name: str
    max_seq_length: int
    word_embedding_dimension: int
    dump_patches: bool


class LargeLanguageModel(BaseModel):
    tokenizer: LLMEncodingTypeDef
    attention_mechanism: LLMAttentionMechanismTypeDef
    embedding: LLMEmbeddingTypeDef
    instance: LLMInstanceTypeDef
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
        json_encoders={
            np.ndarray: LLMEncoder().encode,
            torch.Tensor: LLMEncoder().encode,
            transformers.PreTrainedModel: LLMEncoder().encode,
            transformers.PreTrainedTokenizer: LLMEncoder().encode,
            transformers.PretrainedConfig: LLMEncoder().encode,
            transformers.PreTrainedTokenizerFast: LLMEncoder().encode,
            torch.nn.Module: LLMEncoder().encode,
            torch.optim.Optimizer: LLMEncoder().encode,
        },
    )
