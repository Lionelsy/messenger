"""
配置 AI 模型与参数

- LLM:
    - zhipu (GLM HTTP API)
    - openai_compat (OpenAI-compatible /v1/chat/completions)
- OCR:
    - MinerU (HTTP API)

底部包含简单测试入口
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from urllib.parse import urlsplit, urlunsplit

import requests


# ============================================================
# Config dataclasses
# ============================================================

@dataclass
class ZhipuConfig:
    # 不要把真实 key 写进代码；通过环境变量 ZHIPU_API_KEY 注入
    api_key: str = ""
    model: str = "glm-4.5-flash"
    # 目前走官方 SDK，不再直接用 base_url 发 HTTP；保留字段以便排查/未来扩展
    base_url: str = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    temperature: float = 0.2
    max_tokens: int = 2400
    timeout: int = 300


@dataclass
class OpenAICompatConfig:
    model: str = "gpt-oss:120b"
    # e.g. http://127.0.0.1:8000/v1
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = ""
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 2400
    timeout: int = 300
    # 兼容不同 OpenAI-like 服务的路由习惯；通常是 /v1/chat/completions
    chat_completions_path: str = "/chat/completions"


@dataclass
class MinerUOCRConfig:
    enabled: bool = True
    # e.g. http://127.0.0.1:6001/file_parse
    base_url: str = "http://127.0.0.1:6001/file_parse"
    timeout: int = 600
    # MinerU 的表单字段名可能不是 file；允许通过环境变量覆盖
    file_field: str = "files"
    # 目前按 PDF 使用；如果后续要支持图片/其它类型，可再扩展
    content_type: str = "application/pdf"


@dataclass
class AIConfig:
    llm_provider: str               # "zhipu" | "openai_compat"
    zhipu: Optional[ZhipuConfig]
    openai_compat: Optional[OpenAICompatConfig]
    ocr: MinerUOCRConfig


# ============================================================
# Config loader
# ============================================================

def load_ai_config() -> AIConfig:
    provider = os.getenv("LLM_PROVIDER", "zhipu")

    zhipu_cfg = None
    if provider == "zhipu":
        zhipu_cfg = ZhipuConfig(
            api_key=os.getenv("ZHIPU_API_KEY", ""),
            model=os.getenv("ZHIPU_MODEL", "glm-4.5-flash"),
        )

    openai_cfg = None
    if provider == "openai_compat":
        openai_cfg = OpenAICompatConfig(
            model=os.getenv("LLM_MODEL", "gpt-oss:120b"),
            base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
            api_key=os.getenv("LLM_API_KEY", ""),
            chat_completions_path=os.getenv("LLM_CHAT_PATH", "/chat/completions"),
        )

    ocr_cfg = MinerUOCRConfig(
        enabled=os.getenv("OCR_ENABLED", "True").lower() in ("1", "true", "yes"),
        base_url=os.getenv("MINERU_OCR_URL", "http://127.0.0.1:6001/file_parse"),
        file_field=os.getenv("MINERU_OCR_FILE_FIELD", "files"),
        content_type=os.getenv("MINERU_OCR_CONTENT_TYPE", "application/pdf"),
    )

    return AIConfig(
        llm_provider=provider,
        zhipu=zhipu_cfg,
        openai_compat=openai_cfg,
        ocr=ocr_cfg,
    )


# ============================================================
# LLM Client
# ============================================================

class LLMClient:

    def __init__(self, cfg: AIConfig):
        self.cfg = cfg

    # ---------- public ----------

    def chat(
        self,
        messages: List[Dict[str, str]],
        response_json: bool = False,
    ) -> Dict[str, Any]:
        if self.cfg.llm_provider == "zhipu":
            return self._chat_zhipu(messages, response_json)
        elif self.cfg.llm_provider == "openai_compat":
            return self._chat_openai_compat(messages, response_json)
        else:
            raise ValueError(f"Unknown LLM provider: {self.cfg.llm_provider}")

    def chat_text(self, messages: List[Dict[str, str]], **kwargs) -> str:
        resp = self.chat(messages, **kwargs)
        try:
            return resp["choices"][0]["message"]["content"]
        except Exception:
            return json.dumps(resp, ensure_ascii=False, indent=2)

    # ---------- private ----------

    @staticmethod
    def _openai_compat_chat_url(base_url: str, chat_path: str) -> str:
        """
        兼容两种 base_url 写法：
        - 传入 http://host:port/v1  -> 拼成 /v1/chat/completions
        - 传入 http://host:port     -> 自动补 /v1，再拼 /chat/completions
        """
        b = base_url.rstrip("/")
        p = "/" + chat_path.lstrip("/")
        if b.endswith("/v1"):
            return b + p
        return b + "/v1" + p

    def _chat_openai_compat(
        self,
        messages: List[Dict[str, str]],
        response_json: bool,
    ) -> Dict[str, Any]:
        cfg = self.cfg.openai_compat
        assert cfg is not None

        url = self._openai_compat_chat_url(cfg.base_url, cfg.chat_completions_path)
        headers = {"Content-Type": "application/json"}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"

        payload: Dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "max_tokens": cfg.max_tokens,
        }
        if response_json:
            payload["response_format"] = {"type": "json_object"}

        r = requests.post(url, headers=headers, json=payload, timeout=cfg.timeout)
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            body = (r.text or "").strip()
            if body:
                raise requests.HTTPError(f"{e} | response_body={body[:2000]}") from e
            raise
        return r.json()

    def _chat_zhipu(
        self,
        messages: List[Dict[str, str]],
        response_json: bool,
    ) -> Dict[str, Any]:
        cfg = self.cfg.zhipu
        assert cfg is not None

        # 按智谱官方 SDK 调用（你贴的示例）
        # pip install zhipuai
        from zhipuai import ZhipuAI

        if not cfg.api_key:
            raise RuntimeError("ZHIPU_API_KEY 为空：请通过环境变量设置智谱 API Key")

        client = ZhipuAI(api_key=cfg.api_key)

        kwargs: Dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
        }
        # 智谱 SDK 新版一般兼容 OpenAI 的 response_format；如果你不需要 JSON，可忽略
        if response_json:
            kwargs["response_format"] = {"type": "json_object"}

        resp = client.chat.completions.create(**kwargs)

        # 统一成 dict，兼容现有 chat_text() 的 resp["choices"][0]["message"]["content"]
        if isinstance(resp, dict):
            return resp
        if hasattr(resp, "model_dump"):
            return resp.model_dump()
        if hasattr(resp, "dict"):
            return resp.dict()
        # 兜底：尽量取出 content
        try:
            content = resp.choices[0].message.content  # type: ignore[attr-defined]
            return {"choices": [{"message": {"content": content}}]}
        except Exception:
            return {"raw": str(resp)}


def _mask_secret(s: str, keep: int = 4) -> str:
    if not s:
        return ""
    if len(s) <= keep * 2:
        return "*" * len(s)
    return s[:keep] + "*" * (len(s) - keep * 2) + s[-keep:]


def _safe_cfg_repr(cfg: AIConfig) -> str:
    """
    避免把 API Key 打到终端日志里（你刚才就遇到了泄露风险）。
    """
    data = {
        "llm_provider": cfg.llm_provider,
        "zhipu": None,
        "openai_compat": None,
        "ocr": {
            "enabled": cfg.ocr.enabled,
            "base_url": cfg.ocr.base_url,
            "timeout": cfg.ocr.timeout,
            "file_field": cfg.ocr.file_field,
            "content_type": cfg.ocr.content_type,
        },
    }
    if cfg.zhipu is not None:
        data["zhipu"] = {
            "api_key": _mask_secret(cfg.zhipu.api_key),
            "model": cfg.zhipu.model,
            "base_url": cfg.zhipu.base_url,
            "temperature": cfg.zhipu.temperature,
            "max_tokens": cfg.zhipu.max_tokens,
            "timeout": cfg.zhipu.timeout,
        }
    if cfg.openai_compat is not None:
        data["openai_compat"] = {
            "model": cfg.openai_compat.model,
            "base_url": cfg.openai_compat.base_url,
            "api_key": _mask_secret(cfg.openai_compat.api_key),
            "temperature": cfg.openai_compat.temperature,
            "top_p": cfg.openai_compat.top_p,
            "max_tokens": cfg.openai_compat.max_tokens,
            "timeout": cfg.openai_compat.timeout,
            "chat_completions_path": cfg.openai_compat.chat_completions_path,
        }
    return json.dumps(data, ensure_ascii=False, indent=2)


# ============================================================
# OCR Client (MinerU)
# ============================================================

class OCRClient:

    def __init__(self, cfg: MinerUOCRConfig):
        self.cfg = cfg

    def ocr_pdf(self, pdf_path: str) -> Dict[str, Any]:
        if not self.cfg.enabled:
            raise RuntimeError("OCR is disabled")

        with open(pdf_path, "rb") as f:
            filename = os.path.basename(pdf_path)
            # MinerU 示例：files={"files": ("xxx.pdf", f, "application/pdf")}
            files = {self.cfg.file_field: (filename, f, self.cfg.content_type)}
            r = requests.post(
                self.cfg.base_url,
                files=files,
                timeout=self.cfg.timeout,
            )
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            body = (r.text or "").strip()
            if body:
                raise requests.HTTPError(f"{e} | response_body={body[:2000]}") from e
            raise
        return r.json()


# ============================================================
# Factory
# ============================================================

def get_ai_clients():
    cfg = load_ai_config()
    llm = LLMClient(cfg)
    ocr = OCRClient(cfg.ocr)
    return cfg, llm, ocr


# ============================================================
# Minimal test
# ============================================================

# if __name__ == "__main__":
#     cfg, llm, ocr = get_ai_clients()

#     print("=== AI CONFIG ===")
#     print(_safe_cfg_repr(cfg))

#     print("\n=== LLM TEST ===")
#     messages = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Give one sentence summary of what arXiv is."},
#     ]
#     try:
#         out = llm.chat_text(messages)
#         print(out)
#     except Exception as e:
#         print("LLM test failed:", e)

#     if cfg.ocr.enabled:
#         print("\n=== OCR TEST ===")
#         try:
#             res = ocr.ocr_pdf("test.pdf")
#             print(res.keys())
#         except Exception as e:
#             print("OCR test failed:", e)
