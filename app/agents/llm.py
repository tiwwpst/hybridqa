from __future__ import annotations

import logging
import os
from functools import lru_cache
from types import SimpleNamespace
from typing import Iterable

import torch
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from app.config import Settings

logger = logging.getLogger(__name__)

def _message_content(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, list):
        return " ".join(str(part) for part in content)
    return str(content)


def _messages_to_chat(messages: Iterable[BaseMessage]) -> list[dict]:
    chat = []
    for message in messages:
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            role = "user"
        chat.append({"role": role, "content": _message_content(message)})
    return chat


def _format_prompt(tokenizer, messages: Iterable[BaseMessage]) -> str:
    chat = _messages_to_chat(messages)
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
    lines = []
    for item in chat:
        lines.append(f"{item['role'].upper()}: {item['content']}")
    lines.append("ASSISTANT:")
    return "\n".join(lines)


@lru_cache(maxsize=2)
def _load_hf_pipeline(model_id: str, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer), tokenizer


class LocalHfChatModel:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pipeline, self.tokenizer = _load_hf_pipeline(
            settings.hf_model_id, settings.hf_trust_remote_code
        )

    def invoke(self, messages: Iterable[BaseMessage]):
        prompt = _format_prompt(self.tokenizer, messages)
        outputs = self.pipeline(
            prompt,
            max_new_tokens=self.settings.hf_max_new_tokens,
            do_sample=self.settings.hf_temperature > 0,
            temperature=self.settings.hf_temperature,
            top_p=self.settings.hf_top_p,
        )
        payload = outputs[0] if outputs else {}
        text = payload.get("generated_text") or payload.get("text") or ""
        if text.startswith(prompt):
            text = text[len(prompt) :]
        return SimpleNamespace(content=text.strip())


def build_llm(settings: Settings, model: str):
    if settings.llm_provider == "openai" and settings.openai_api_key:
        json_mode = settings.openai_json_mode
        if json_mode:
            try:
                return ChatOpenAI(
                    model=model,
                    api_key=settings.openai_api_key,
                    model_kwargs={"response_format": {"type": "json_object"}},
                )
            except Exception as exc:
                logger.warning("OpenAI init failed (json_mode): %s", exc)
        try:
            return ChatOpenAI(model=model, api_key=settings.openai_api_key)
        except Exception as exc:
            logger.warning("OpenAI init failed: %s", exc)
            return None
    if settings.hf_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        os.environ.setdefault("HF_HUB_OFFLINE", "0")
    try:
        return LocalHfChatModel(settings)
    except Exception as exc:
        logger.warning("HF model init failed: %s", exc)
        return None
