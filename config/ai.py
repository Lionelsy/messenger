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

import io
import os
import json
import re
import time
import random
import zipfile
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

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
    base_url: str = "https://open.bigmodel.cn/api/coding/paas/v4"
    temperature: float = 0.2
    max_tokens: int = 8192
    timeout: int = 1000


@dataclass
class OpenAICompatConfig:
    model: str = "gpt-oss:120b"
    # e.g. http://127.0.0.1:8000/v1
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = ""
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 8192
    timeout: int = 1000
    # 兼容不同 OpenAI-like 服务的路由习惯；通常是 /v1/chat/completions
    chat_completions_path: str = "/chat/completions"
    # 思考模式控制：""=保持模型默认（推荐）；"off"=禁用思考；"on"=强制开启
    # vLLM/SGLang 约定：chat_template_kwargs.enable_thinking
    thinking_control: str = ""


@dataclass
class MinerUOCRConfig:
    """OCR 客户端配置：支持双后端，通过 provider 二选一（纯配置切换，不自动回退）。

    - cloud: MinerU 官方 API (mineru.net)，异步"申请链接→上传→轮询→下载 zip"，默认
    - local:  自建 MinerU HTTP 服务（原逻辑）
    """
    enabled: bool = True
    # "cloud"(mineru.net 官方 API) | "local"(自建 HTTP)
    provider: str = "cloud"

    # ---- local：自建 MinerU HTTP（原字段，语义不变）----
    # e.g. http://127.0.0.1:6001/file_parse
    base_url: str = "http://127.0.0.1:6001/file_parse"
    timeout: int = 600
    # MinerU 的表单字段名可能不是 file；允许通过环境变量覆盖
    file_field: str = "files"
    # 目前按 PDF 使用；如果后续要支持图片/其它类型，可再扩展
    content_type: str = "application/pdf"

    # ---- cloud：MinerU 官方 API (mineru.net/api/v4) ----
    api_key: str = ""                       # MINERU_KEY，Authorization: Bearer <key>
    cloud_base_url: str = "https://mineru.net/api/v4"
    cloud_model: str = "pipeline"           # "pipeline" | "vlm"
    cloud_language: str = "en"              # "ch" | "en"；arXiv 论文建议 en
    cloud_enable_formula: bool = True
    cloud_enable_table: bool = True
    cloud_is_ocr: bool = False              # per-file is_ocr；扫描版 PDF 可开
    cloud_poll_interval: float = 5.0        # 状态轮询间隔（秒）
    cloud_timeout: float = 600.0            # 轮询总超时（墙钟秒）
    cloud_http_timeout: float = 180.0       # 单次 HTTP 调用超时（申请/上传/下载）
    cloud_retries: int = 2                  # 瞬态错误重试次数（申请/上传/下载各自独立）


@dataclass
class AIConfig:
    llm_provider: str               # "zhipu" | "openai_compat"
    zhipu: Optional[ZhipuConfig]
    openai_compat: Optional[OpenAICompatConfig]
    ocr: MinerUOCRConfig


# ============================================================
# Config loader
# ============================================================

def _env_int(name: str, default: int) -> int:
    """安全读取整型环境变量；非法值回退默认值，避免流水线启动即崩溃。"""
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        print(f"[WARN] {name}={os.getenv(name)!r} 不是合法整数，使用默认值 {default}", flush=True)
        return default


def _load_thinking_control() -> str:
    """读取 LLM_THINKING：合法值 ""/off/on；非法值警告并回退默认（保持模型思考模式）。"""
    v = os.getenv("LLM_THINKING", "").strip().lower()
    if v not in ("", "off", "on"):
        print(f"[WARN] LLM_THINKING={os.getenv('LLM_THINKING')!r} 非法，回退为空（保持模型默认）", flush=True)
        return ""
    return v


def _env_float(name: str, default: float) -> float:
    """安全读取浮点环境变量；非法值回退默认值，与 _env_int 同风格。"""
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        print(f"[WARN] {name}={os.getenv(name)!r} 不是合法数字，使用默认值 {default}", flush=True)
        return default


def _load_ocr_provider() -> str:
    """读取 OCR_PROVIDER：合法值 cloud/local；非法值警告并回退 cloud。"""
    v = os.getenv("OCR_PROVIDER", "cloud").strip().lower()
    if v not in ("cloud", "local"):
        print(f"[WARN] OCR_PROVIDER={os.getenv('OCR_PROVIDER')!r} 非法，回退为 cloud", flush=True)
        return "cloud"
    return v


def _load_cloud_model() -> str:
    """读取 MINERU_CLOUD_MODEL：pipeline/vlm；非法值警告并回退 pipeline。"""
    v = os.getenv("MINERU_CLOUD_MODEL", "pipeline").strip().lower()
    if v not in ("pipeline", "vlm"):
        print(f"[WARN] MINERU_CLOUD_MODEL={os.getenv('MINERU_CLOUD_MODEL')!r} 非法，回退为 pipeline", flush=True)
        return "pipeline"
    return v


def _load_cloud_language() -> str:
    """读取 MINERU_CLOUD_LANGUAGE：ch/en；非法值警告并回退 en（arXiv 论文为英文）。"""
    v = os.getenv("MINERU_CLOUD_LANGUAGE", "en").strip().lower()
    if v not in ("ch", "en"):
        print(f"[WARN] MINERU_CLOUD_LANGUAGE={os.getenv('MINERU_CLOUD_LANGUAGE')!r} 非法，回退为 en", flush=True)
        return "en"
    return v


def load_ai_config() -> AIConfig:
    provider = os.getenv("LLM_PROVIDER", "zhipu")

    # 思考模式保留时，max_tokens 同时覆盖"思考+答案"，太小会被思考吃掉导致答案截断
    # 默认 8192；如后端上下文较小（max-model-len 小），需同步调低
    max_tokens = _env_int("LLM_MAX_TOKENS", 8192)

    zhipu_cfg = None
    if provider == "zhipu":
        zhipu_cfg = ZhipuConfig(
            api_key=os.getenv("ZHIPU_API_KEY", ""),
            model=os.getenv("ZHIPU_MODEL", "glm-4.5-flash"),
            max_tokens=_env_int("ZHIPU_MAX_TOKENS", 2400),
        )

    openai_cfg = None
    if provider == "openai_compat":
        openai_cfg = OpenAICompatConfig(
            model=os.getenv("LLM_MODEL", "gpt-oss:120b"),
            base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
            api_key=os.getenv("LLM_API_KEY", ""),
            max_tokens=max_tokens,
            chat_completions_path=os.getenv("LLM_CHAT_PATH", "/chat/completions"),
            thinking_control=_load_thinking_control(),
        )

    ocr_cfg = MinerUOCRConfig(
        enabled=os.getenv("OCR_ENABLED", "True").lower() in ("1", "true", "yes"),
        provider=_load_ocr_provider(),
        # —— local：自建 MinerU HTTP ——
        base_url=os.getenv("MINERU_OCR_URL", "http://127.0.0.1:6001/file_parse"),
        file_field=os.getenv("MINERU_OCR_FILE_FIELD", "files"),
        content_type=os.getenv("MINERU_OCR_CONTENT_TYPE", "application/pdf"),
        # —— cloud：MinerU 官方 API ——
        api_key=os.getenv("MINERU_KEY", ""),
        cloud_base_url=os.getenv("MINERU_CLOUD_BASE_URL", "https://mineru.net/api/v4"),
        cloud_model=_load_cloud_model(),
        cloud_language=_load_cloud_language(),
        cloud_enable_formula=os.getenv("MINERU_CLOUD_FORMULA", "true").lower() in ("1", "true", "yes"),
        cloud_enable_table=os.getenv("MINERU_CLOUD_TABLE", "true").lower() in ("1", "true", "yes"),
        cloud_is_ocr=os.getenv("MINERU_CLOUD_IS_OCR", "false").lower() in ("1", "true", "yes"),
        cloud_poll_interval=_env_float("MINERU_CLOUD_POLL_INTERVAL", 5.0),
        cloud_timeout=_env_float("MINERU_CLOUD_TIMEOUT", 600.0),
        cloud_http_timeout=_env_float("MINERU_CLOUD_HTTP_TIMEOUT", 180.0),
        cloud_retries=_env_int("MINERU_CLOUD_RETRIES", 2),
    )

    # cloud 模式但 key 为空：仅提示（调用时才会真正报错），避免启动即崩溃
    if ocr_cfg.provider == "cloud" and not ocr_cfg.api_key:
        print(
            "[WARN] OCR_PROVIDER=cloud 但 MINERU_KEY 为空：请设置官方 API Key，否则解析将全部失败",
            flush=True,
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
            choice = resp["choices"][0]
            finish_reason = choice.get("finish_reason")
            msg = choice["message"]
            content = msg.get("content")
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"LLM 响应结构异常: {e}; keys={list(resp.keys())}") from e
        if not content:
            # 思考模型把 CoT 放在 reasoning/reasoning_content，但只有 content 才是最终答案；
            # 只拿到思考（尤其 finish_reason=length 时）说明输出被截断，必须失败让该篇重试
            raise ValueError(f"LLM 未返回最终答案（content 为空, finish_reason={finish_reason!r}）")
        if finish_reason == "length":
            # 思考保留时，CoT 可能占用大量预算导致 length：内容非空说明已有部分答案，
            # 警告并返回部分内容，避免整个阶段卡死；内容为空则上面已抛错
            print(
                f"[WARN] LLM 输出被截断(finish_reason=length), 返回部分内容; "
                f"可调大 LLM_MAX_TOKENS 或检查 --max-model-len",
                flush=True,
            )
        text = _strip_thinking_content(content)
        if not text.strip():
            raise ValueError("LLM 输出剥离思考内容后为空")
        return text

    # ---------- private ----------

    @staticmethod
    def _openai_compat_chat_url(base_url: str, chat_path: str) -> str:
        """
        兼容多种 base_url 写法：
        - 传入 http://host:port/v1  -> /v1/chat/completions
        - 传入 http://host:port/v4  -> /v4/chat/completions
        - 传入 http://host:port     -> 自动补 /v1，再拼 /chat/completions
        """
        b = base_url.rstrip("/")
        p = "/" + chat_path.lstrip("/")
        if re.search(r"/v\d+$", b):
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
        # 思考模式控制（vLLM/SGLang 约定；""=保持模型默认，不发送任何控制字段）
        if cfg.thinking_control == "off":
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        elif cfg.thinking_control == "on":
            payload["chat_template_kwargs"] = {"enable_thinking": True}
        if response_json:
            payload["response_format"] = {"type": "json_object"}

        max_retries = 5
        backoff_base = 5.0
        last_err: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=cfg.timeout)
                r.raise_for_status()
                return r.json()
            except requests.HTTPError as e:
                last_err = e
                status = e.response.status_code if e.response is not None else None
                if status in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    sleep_s = backoff_base * (2 ** attempt) + random.uniform(0, 2)
                    body = (e.response.text or "").strip()[:200] if e.response is not None else ""
                    print(
                        f"[WARN] LLM HTTP {status}; retry {attempt+1}/{max_retries} in {sleep_s:.1f}s | {body}",
                        flush=True,
                    )
                    time.sleep(sleep_s)
                else:
                    body = (e.response.text or "").strip()[:2000] if e.response is not None else ""
                    if body:
                        raise requests.HTTPError(f"{e} | response_body={body}") from e
                    raise
            except (requests.ConnectionError, requests.Timeout) as e:
                last_err = e
                if attempt < max_retries - 1:
                    sleep_s = backoff_base * (2 ** attempt) + random.uniform(0, 2)
                    print(
                        f"[WARN] LLM {type(e).__name__}; retry {attempt+1}/{max_retries} in {sleep_s:.1f}s",
                        flush=True,
                    )
                    time.sleep(sleep_s)
                else:
                    raise

        if last_err is not None:
            raise last_err
        raise RuntimeError("LLM request failed after retries")

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


def _strip_thinking_content(text: str) -> str:
    """
    过滤模型输出中的“思考/推理”内容，只保留最终答案部分。

    兼容常见形式：
    - <thinking>...</thinking> + 最终答案
    - 只有闭合标签（部分后端会把思考段落直接拼在 content 前，再补一个闭合标签）
    - 多段 <thinking>...</thinking>（全部移除）
    - 未闭合的 <thinking>（输出被截断）→ 返回空，交由上层按失败处理
    """
    OPEN = chr(60) + "thinking" + chr(62)      # <thinking>
    CLOSE = chr(60) + "/thinking" + chr(62)    # </thinking>

    t = (text or "")
    if not t:
        return ""

    # 1) 标准成对标签：直接剔除所有 think 块
    if OPEN in t and CLOSE in t:
        t2 = re.sub(re.escape(OPEN) + r"[\s\S]*?" + re.escape(CLOSE), "", t, flags=re.IGNORECASE).strip()
        return t2

    # 2) 只有闭合标签：保留最后一个闭合标签之后的内容
    if CLOSE in t:
        tail = t.rsplit(CLOSE, 1)[-1].strip()
        return tail

    # 3) 只有开启标签（截断流）：剩余内容全部是思考，返回空
    if OPEN in t:
        return ""

    return t.strip()


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
            "provider": cfg.ocr.provider,
            "base_url": cfg.ocr.base_url,
            "timeout": cfg.ocr.timeout,
            "file_field": cfg.ocr.file_field,
            "content_type": cfg.ocr.content_type,
            "api_key": _mask_secret(cfg.ocr.api_key),
            "cloud_base_url": cfg.ocr.cloud_base_url,
            "cloud_model": cfg.ocr.cloud_model,
            "cloud_language": cfg.ocr.cloud_language,
            "cloud_enable_formula": cfg.ocr.cloud_enable_formula,
            "cloud_enable_table": cfg.ocr.cloud_enable_table,
            "cloud_is_ocr": cfg.ocr.cloud_is_ocr,
            "cloud_poll_interval": cfg.ocr.cloud_poll_interval,
            "cloud_timeout": cfg.ocr.cloud_timeout,
            "cloud_http_timeout": cfg.ocr.cloud_http_timeout,
            "cloud_retries": cfg.ocr.cloud_retries,
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
            "thinking_control": cfg.openai_compat.thinking_control,
        }
    return json.dumps(data, ensure_ascii=False, indent=2)


# ============================================================
# OCR Client (MinerU)
# ============================================================

class OCRClient:
    """MinerU OCR 客户端。

    provider=cloud -> MinerU 官方 API（mineru.net），异步：申请上传链接 → PUT 上传 → 轮询 → 下载 zip 取 full.md
    provider=local -> 自建 MinerU HTTP 服务（原逻辑）

    任何失败一律 raise（绝不构造伪成功结果），由调用方标记 download=False 以便重跑重试。
    """

    # 业务错误码：瞬态（排队等），可退避重试
    _CLOUD_RETRYABLE_CODES = {"-60001", "-60007", "-60009", "-60010", "-10001"}
    # HTTP 状态码：瞬态
    _CLOUD_RETRYABLE_HTTP = {429, 500, 502, 503, 504}
    _CLOUD_MAX_FILE_BYTES = 200 * 1024 * 1024  # 云端单文件上限 200MB

    def __init__(self, cfg: MinerUOCRConfig):
        self.cfg = cfg

    def ocr_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """按 OCR_PROVIDER 分发；任何失败一律 raise（不再构造 error-fallback）。"""
        if not self.cfg.enabled:
            raise RuntimeError("OCR is disabled (OCR_ENABLED=False)")
        if self.cfg.provider == "cloud":
            return self._ocr_pdf_cloud(pdf_path)
        if self.cfg.provider == "local":
            return self._ocr_pdf_local(pdf_path)
        raise ValueError(f"Unknown OCR provider: {self.cfg.provider!r}")

    # ---------- local：自建 MinerU HTTP ----------

    def _ocr_pdf_local(self, pdf_path: str) -> Dict[str, Any]:
        """自建 MinerU HTTP：原逻辑，仅删除 error-fallback，失败直接抛。"""
        filename = os.path.basename(pdf_path)

        with open(pdf_path, "rb") as f:
            files = {self.cfg.file_field: (filename, f, self.cfg.content_type)}
            r = requests.post(
                self.cfg.base_url,
                files=files,
                timeout=self.cfg.timeout,
            )

        r.raise_for_status()
        data = r.json()
        if not (data.get("results") or {}):
            raise RuntimeError(f"[local] MinerU 响应缺少 results: {str(data)[:200]}")
        return data

    # ---------- cloud：MinerU 官方 API ----------

    def _ocr_pdf_cloud(self, pdf_path: str) -> Dict[str, Any]:
        cfg = self.cfg
        filename = os.path.basename(pdf_path)
        paper_id = os.path.splitext(filename)[0]  # 不带 .pdf，保证下游 results[paper_id] 命中

        if not cfg.api_key:
            raise RuntimeError("MINERU_KEY 为空：OCR_PROVIDER=cloud 需要官方 API Key")

        size = os.path.getsize(pdf_path)
        if size > self._CLOUD_MAX_FILE_BYTES:
            raise ValueError(
                f"PDF 超过 MinerU 云端 200MB 限制 ({size / 1024 / 1024:.1f}MB)"
            )

        headers = {"Authorization": f"Bearer {cfg.api_key}"}  # PUT 时也不带 Content-Type（文档要求）
        base = cfg.cloud_base_url.rstrip("/")
        t0 = time.time()

        # ---- 第 1 步：申请上传链接 ----
        body = {
            "model_version": cfg.cloud_model,
            "language": cfg.cloud_language,
            "enable_formula": cfg.cloud_enable_formula,
            "enable_table": cfg.cloud_enable_table,
            # 不传 callback/seed -> 只能轮询
            "files": [
                {"name": filename, "data_id": paper_id, "is_ocr": cfg.cloud_is_ocr},
            ],
        }
        batch_id, file_urls = self._cloud_post_batch(base, headers, body, paper_id)

        # ---- 第 2 步：PUT 上传文件 ----
        self._cloud_put_upload(file_urls[0], pdf_path, headers, paper_id)

        # ---- 第 3 步：轮询解析状态 ----
        zip_url = self._cloud_poll_batch(base, headers, batch_id, paper_id, filename)

        # ---- 第 4 步：下载 zip 并取 full.md ----
        md = self._cloud_fetch_md(zip_url, paper_id)

        print(
            f"[INFO] {paper_id} MinerU 云端解析完成（{len(md)} 字符, 耗时 {time.time() - t0:.1f}s）",
            flush=True,
        )
        return {
            "backend": "mineru-cloud",
            "version": cfg.cloud_model,  # 仅记录用途，无消费方
            "results": {paper_id: {"md_content": md}},
        }

    def _cloud_post_batch(
        self, base: str, headers: Dict[str, str], body: Dict[str, Any], paper_id: str
    ) -> Tuple[str, List[str]]:
        """POST /file-urls/batch 申请上传链接；返回 (batch_id, file_urls)。"""
        cfg = self.cfg
        url = f"{base}/file-urls/batch"

        for attempt in range(cfg.cloud_retries + 1):
            try:
                r = requests.post(url, json=body, headers=headers, timeout=cfg.cloud_http_timeout)
                if r.status_code in self._CLOUD_RETRYABLE_HTTP and attempt < cfg.cloud_retries:
                    raise _RetrySignal(f"HTTP {r.status_code}", r)

                r.raise_for_status()
                j = r.json()
                code = str(j.get("code"))
                if code != "0":
                    msg = j.get("msg") or ""
                    trace_id = j.get("trace_id") or ""
                    if code in self._CLOUD_RETRYABLE_CODES and attempt < cfg.cloud_retries:
                        raise _RetrySignal(f"code={code} msg={msg}")
                    if code in ("A0202", "A0211"):
                        raise RuntimeError(f"MinerU Token 无效或过期（{code}），请检查 MINERU_KEY")
                    if code == "-60005":
                        raise RuntimeError("文件超过 MinerU 云端 200MB 限制（-60005）")
                    if code == "-60006":
                        raise RuntimeError("文件超过 MinerU 云端 200 页限制（-60006）")
                    if code == "-60018":
                        raise RuntimeError(
                            "今日 MinerU 云端任务配额已达上限（-60018），请明天再试或调低处理量"
                        )
                    raise RuntimeError(f"申请上传链接失败 code={code} msg={msg} trace_id={trace_id}")

                data = j.get("data") or {}
                batch_id = data.get("batch_id") or ""
                file_urls = data.get("file_urls") or []
                if not batch_id or not file_urls:
                    raise RuntimeError(f"申请上传链接响应缺少 batch_id/file_urls: {str(j)[:300]}")
                if len(file_urls) != len(body.get("files") or []):
                    raise RuntimeError(
                        f"申请上传链接数量不匹配: files={len(body.get('files') or [])} urls={len(file_urls)}"
                    )
                if not str(file_urls[0]).startswith("http"):
                    raise RuntimeError(f"上传链接异常: {file_urls[0]!r}")
                return batch_id, list(file_urls)

            except _RetrySignal as sig:
                sleep_s = self._cloud_backoff(attempt, getattr(sig, "resp", None))
                print(
                    f"[WARN] {paper_id} MinerU 申请链接失败（{sig}）；重试 {attempt + 1}/{cfg.cloud_retries} 后 {sleep_s:.1f}s",
                    flush=True,
                )
                time.sleep(sleep_s)
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt < cfg.cloud_retries:
                    sleep_s = self._cloud_backoff(attempt)
                    print(
                        f"[WARN] {paper_id} MinerU 申请链接 {type(e).__name__}；重试 {attempt + 1}/{cfg.cloud_retries} 后 {sleep_s:.1f}s",
                        flush=True,
                    )
                    time.sleep(sleep_s)
                else:
                    raise

        # 理论不可达（循环内最后一次尝试的异常都会直接传播）；保留作类型安全兜底
        raise RuntimeError("申请上传链接失败（重试耗尽）")

    def _cloud_put_upload(
        self, url: str, pdf_path: str, headers: Dict[str, str], paper_id: str
    ) -> None:
        """PUT 上传文件到签名链接；返回 None（HTTP 200 即成功）。

        文档要求：上传时无须设置 Content-Type 请求头（headers 只带 Authorization）。
        """
        cfg = self.cfg

        for attempt in range(cfg.cloud_retries + 1):
            try:
                # 每次重试重新 open，避免文件指针耗尽
                with open(pdf_path, "rb") as f:
                    r = requests.put(url, data=f, headers=headers, timeout=cfg.cloud_http_timeout)

                if r.status_code == 200:
                    return
                if r.status_code in (403, 404):
                    # 签名链接失效/过期（24h 有效期），重试同一个坏 URL 无意义
                    raise RuntimeError(f"上传链接失效（HTTP {r.status_code}），请重跑重新申请")
                if r.status_code in self._CLOUD_RETRYABLE_HTTP and attempt < cfg.cloud_retries:
                    raise _RetrySignal(r.status_code)
                raise RuntimeError(f"PDF 上传失败（HTTP {r.status_code}）: {r.text[:200]}")

            except _RetrySignal as sig:
                sleep_s = self._cloud_backoff(attempt)
                print(
                    f"[WARN] {paper_id} PUT 上传失败（{sig}）；重试 {attempt + 1}/{cfg.cloud_retries} 后 {sleep_s:.1f}s",
                    flush=True,
                )
                time.sleep(sleep_s)
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt < cfg.cloud_retries:
                    sleep_s = self._cloud_backoff(attempt)
                    print(
                        f"[WARN] {paper_id} PUT 上传 {type(e).__name__}；重试 {attempt + 1}/{cfg.cloud_retries} 后 {sleep_s:.1f}s",
                        flush=True,
                    )
                    time.sleep(sleep_s)
                else:
                    raise

        # 理论不可达（循环内最后一次尝试的异常都会直接传播）；保留作类型安全兜底
        raise RuntimeError(f"PDF 上传失败（重试耗尽）: {url}")

    def _cloud_poll_batch(
        self, base: str, headers: Dict[str, str], batch_id: str, paper_id: str, filename: str
    ) -> str:
        """轮询 batch 状态直到 done；返回 full_zip_url。"""
        cfg = self.cfg
        url = f"{base}/extract-results/batch/{batch_id}"
        deadline = time.monotonic() + cfg.cloud_timeout
        last_state = "?"
        consecutive_fails = 0
        last_prog: Optional[Tuple] = None

        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"MinerU 云端解析超时（{cfg.cloud_timeout:.0f}s），最后状态={last_state}；"
                    f"可调大 MINERU_CLOUD_TIMEOUT 或重跑"
                )

            try:
                r = requests.get(url, headers=headers, timeout=cfg.cloud_http_timeout)
            except (requests.ConnectionError, requests.Timeout) as e:
                consecutive_fails += 1
                if consecutive_fails > 3:
                    raise RuntimeError(f"轮询连续失败 {consecutive_fails} 次: {e}")
                print(f"[WARN] {paper_id} MinerU 轮询网络异常，继续等待: {e}", flush=True)
                time.sleep(min(cfg.cloud_poll_interval, 15))
                continue

            if r.status_code in self._CLOUD_RETRYABLE_HTTP:
                consecutive_fails += 1
                if consecutive_fails > 3:
                    raise RuntimeError(f"轮询连续 HTTP {r.status_code} 失败")
                sleep_s = min(float(r.headers.get("Retry-After") or cfg.cloud_poll_interval), 60)
                print(
                    f"[WARN] {paper_id} MinerU 轮询 HTTP {r.status_code}；{sleep_s:.0f}s 后继续",
                    flush=True,
                )
                time.sleep(sleep_s)
                continue

            r.raise_for_status()
            consecutive_fails = 0
            j = r.json()
            if str(j.get("code")) != "0":
                raise RuntimeError(f"MinerU 轮询错误 code={j.get('code')} msg={j.get('msg')}")

            data = j.get("data") or {}
            items = data.get("extract_result") or []
            item = self._pick_extract_item(items, paper_id, filename)
            # per-file state 优先，batch 级 state 兜底
            state = (item or {}).get("state") or data.get("state") or ""
            last_state = state

            if state == "done":
                zip_url = (item or {}).get("full_zip_url") or ""
                if not zip_url:
                    raise RuntimeError(f"state=done 但缺少 full_zip_url（data_id={paper_id}）")
                return zip_url
            if state == "failed":
                err = (item or {}).get("err_msg") or data.get("err_msg") or "未知错误"
                raise RuntimeError(f"MinerU 云端解析失败: {err}")

            # waiting-file/pending/running/converting/空 -> 继续轮询
            if state == "running":
                prog = (item or {}).get("extract_progress") or {}
                p, t = prog.get("extracted_pages"), prog.get("total_pages")
                cur = (p, t) if (p is not None and t is not None) else None
                if cur and cur != last_prog:
                    print(f"[INFO] {paper_id} MinerU 解析进度 {p}/{t} 页", flush=True)
                    last_prog = cur

            time.sleep(cfg.cloud_poll_interval)

    def _cloud_fetch_md(self, zip_url: str, paper_id: str) -> str:
        """下载 full_zip_url 并从 zip 中提取 full.md 内容。"""
        cfg = self.cfg

        for attempt in range(cfg.cloud_retries + 1):
            try:
                r = requests.get(zip_url, timeout=cfg.cloud_http_timeout)
                r.raise_for_status()
                zf = zipfile.ZipFile(io.BytesIO(r.content))
                names = [n for n in zf.namelist() if n == "full.md" or n.endswith("/full.md")]
                if not names:
                    raise RuntimeError(
                        f"zip 内未找到 full.md，实际文件: {zf.namelist()[:20]}"
                    )
                # 兼容根目录/子目录：取路径最短的 full.md
                target = sorted(names, key=len)[0]
                md = zf.read(target).decode("utf-8", errors="replace")
                if not md.strip():
                    raise RuntimeError(
                        "full.md 内容为空（可能是扫描版 PDF，可设 MINERU_CLOUD_IS_OCR=true）"
                    )
                return md
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt < cfg.cloud_retries:
                    sleep_s = self._cloud_backoff(attempt)
                    print(
                        f"[WARN] {paper_id} 下载解析结果 {type(e).__name__}；重试 {attempt + 1}/{cfg.cloud_retries} 后 {sleep_s:.1f}s",
                        flush=True,
                    )
                    time.sleep(sleep_s)
                else:
                    raise

        raise RuntimeError(f"下载解析结果失败（重试耗尽）: {zip_url}")

    # ---------- 辅助 ----------

    @staticmethod
    def _cloud_backoff(attempt: int, resp: Optional[Any] = None) -> float:
        """瞬态错误退避：2s * 2^attempt + 抖动；优先尊重 Retry-After（cap 60s）。"""
        if resp is not None:
            try:
                ra = float(resp.headers.get("Retry-After") or 0)
                if ra > 0:
                    return min(ra, 60)
            except Exception:
                pass
        return 2.0 * (2 ** attempt) + random.uniform(0, 1)

    @staticmethod
    def _pick_extract_item(
        items: List[Dict[str, Any]], paper_id: str, filename: str
    ) -> Optional[Dict[str, Any]]:
        """从 extract_result 中挑出当前论文对应项：
        优先 data_id == paper_id -> file_name == filename -> 单元素 -> [WARN] 取首项。"""
        if not items:
            return None
        for it in items:
            if (it.get("data_id") or "") == paper_id:
                return it
        for it in items:
            if (it.get("file_name") or "") == filename:
                return it
        if len(items) == 1:
            return items[0]
        print(
            f"[WARN] extract_result 含 {len(items)} 项且无法匹配 data_id={paper_id!r}/file_name={filename!r}，取首项",
            flush=True,
        )
        return items[0]


class _RetrySignal(Exception):
    """内部信号：请求满足瞬态条件，可以退避重试（携带原因与可选的响应对象）。"""

    def __init__(self, reason: Any, resp: Optional[Any] = None):
        super().__init__(str(reason))
        self.reason = reason
        self.resp = resp


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

if __name__ == "__main__":
    import sys
    from pathlib import Path

    cfg, llm, ocr = get_ai_clients()

    print("=== AI CONFIG ===")
    print(_safe_cfg_repr(cfg))

    print("\n=== LLM TEST ===")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Give one sentence summary of what arXiv is."},
    ]
    try:
        out = llm.chat_text(messages)
        print(out)
    except Exception as e:
        print("LLM test failed:", e)

    if cfg.ocr.enabled:
        print("\n=== OCR TEST ===")
        # 用法: python config/ai.py [pdf_path]（默认取 storage/papers/pdfs/ 下第一个 PDF）
        _root = Path(__file__).resolve().parents[1]
        pdf = sys.argv[1] if len(sys.argv) > 1 else ""
        if not pdf or not Path(pdf).exists():
            pdfs = sorted((_root / "storage" / "papers" / "pdfs").glob("*.pdf"))
            pdf = str(pdfs[0]) if pdfs else "test.pdf"
        try:
            res = ocr.ocr_pdf(pdf)
            _rk = list((res.get("results") or {}).keys())
            print("OCR OK | backend:", res.get("backend"), "| results keys:", _rk)
        except Exception as e:
            print("OCR test failed:", e)
