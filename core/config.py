"""配置加载：把 .env 与系统环境变量集中成类型化配置对象。

统一配置入口：换模型 / 换 Key 只改 .env，不改业务代码。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM 连接配置（字段与 .env 一一对应）。"""

    provider: str = Field(default="openai", description="模型服务商，仅作标识")
    model_id: str = Field(default="gpt-4o-mini", description="模型 id")
    api_key: str = Field(default="", description="API Key")
    base_url: str = Field(default="https://api.openai.com/v1", description="OpenAI 兼容接口地址")
    timeout_seconds: float = Field(default=60.0, description="HTTP 请求超时（秒）")
    temperature: float = Field(default=0.2, description="采样温度，越低越稳定")
    max_retries: int = Field(default=2, description="失败重试次数上限")
    # 重试退避基数（秒），第 n 次等待 base * 2^n
    retry_backoff_seconds: float = Field(default=1.0, description="重试退避基数（秒）")
    # 流式请求是否附带 stream_options.include_usage：让服务端在最后一块返回 usage，
    # 否则流式模式无法观测 token 用量与前缀缓存命中（个别网关不支持时可关掉）
    stream_include_usage: bool = Field(default=True, description="流式请求是否请求 usage")


class EmbeddingConfig(BaseModel):
    """Embedding 服务配置（独立于 LLM：服务商可以不同，如 LLM 用 DeepSeek、向量用 SiliconFlow）。"""

    provider: str = Field(default="siliconflow", description="Embedding 服务商标识")
    model_id: str = Field(default="BAAI/bge-m3", description="向量模型 id")
    api_key: str = Field(default="", description="Embedding API Key")
    base_url: str = Field(
        default="https://api.siliconflow.cn/v1", description="OpenAI 兼容接口地址"
    )
    timeout_seconds: float = Field(default=60.0, description="HTTP 请求超时（秒）")
    max_retries: int = Field(default=2, description="失败重试次数上限")
    retry_backoff_seconds: float = Field(default=1.0, description="重试退避基数（秒）")
    max_batch: int = Field(default=32, description="单次请求最大文本数（批量向量化）")


class RAGConfig(BaseModel):
    """RAG 索引参数（分块与文件大小限制）。"""

    chunk_size: int = Field(default=800, description="分块目标大小（字符）")
    chunk_overlap: int = Field(default=100, description="相邻块重叠字符数")
    index_max_chars: int = Field(default=200_000, description="单文件最大索引字符数")


def load_llm_config(env_file: str | Path = ".env") -> LLMConfig:
    """从 .env（若存在）读取配置并返回类型化对象。

    显式传入 env_file 便于测试时指向临时文件。
    """
    env_path = Path(env_file)
    if env_path.is_file():
        load_dotenv(env_path)
    else:
        load_dotenv()  # 默认从当前工作目录找 .env
    return LLMConfig(
        provider=os.getenv("LLM_PROVIDER", "openai"),
        model_id=os.getenv("LLM_MODEL_ID", "gpt-4o-mini"),
        api_key=os.getenv("LLM_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        timeout_seconds=float(os.getenv("LLM_TIMEOUT", "60")),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
        retry_backoff_seconds=float(os.getenv("LLM_RETRY_BACKOFF", "1.0")),
        stream_include_usage=os.getenv("LLM_STREAM_INCLUDE_USAGE", "1") in ("1", "true", "True"),
    )


def load_embedding_config(env_file: str | Path = ".env") -> EmbeddingConfig:
    """从 .env 读取 Embedding 服务配置。

    与 LLM 配置相互独立：不同服务商的接口地址 / Key / 模型可分别设置，
    更换 Embedding 服务商只需改 .env，无需改代码。
    """
    env_path = Path(env_file)
    if env_path.is_file():
        load_dotenv(env_path)
    else:
        load_dotenv()
    return EmbeddingConfig(
        provider=os.getenv("EMBEDDING_PROVIDER", "siliconflow"),
        model_id=os.getenv("EMBEDDING_MODEL_ID", "BAAI/bge-m3"),
        api_key=os.getenv("EMBEDDING_API_KEY", ""),
        base_url=os.getenv("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1"),
        timeout_seconds=float(os.getenv("EMBEDDING_TIMEOUT", "60")),
        max_retries=int(os.getenv("EMBEDDING_MAX_RETRIES", "2")),
        retry_backoff_seconds=float(os.getenv("EMBEDDING_RETRY_BACKOFF", "1.0")),
        max_batch=int(os.getenv("EMBEDDING_MAX_BATCH", "32")),
    )


def load_rag_config(env_file: str | Path = ".env") -> RAGConfig:
    """从 .env 读取 RAG 索引参数（分块大小 / overlap / 单文件上限）。"""
    env_path = Path(env_file)
    if env_path.is_file():
        load_dotenv(env_path)
    else:
        load_dotenv()
    return RAGConfig(
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "100")),
        index_max_chars=int(os.getenv("RAG_INDEX_MAX_CHARS", "200000")),
    )


class ContextConfig(BaseModel):
    """上下文工程参数（L1/L2/L3 拼装与水位 compact）。"""

    max_tokens: int = Field(default=32000, description="上下文预算上限（估算 token）")
    compact_ratio: float = Field(default=0.8, description="水位阈值比例：估算达预算该比例即压缩")
    keep_turns: int = Field(default=20, description="compact 后保留的最近对话单元数")
    project_files: bool = Field(default=True, description="是否读取 AGENTS.md 等项目规则")
    repo_map: bool = Field(default=True, description="是否注入文件地图")
    repo_map_max_lines: int = Field(default=150, description="文件地图最大行数")


def load_context_config(env_file: str | Path = ".env") -> ContextConfig:
    """从 .env 读取上下文工程参数（与 load_llm_config 共用同一环境文件）。"""
    env_path = Path(env_file)
    if env_path.is_file():
        load_dotenv(env_path)
    else:
        load_dotenv()
    return ContextConfig(
        max_tokens=int(os.getenv("CONTEXT_MAX_TOKENS", "32000")),
        compact_ratio=float(os.getenv("CONTEXT_COMPACT_RATIO", "0.8")),
        keep_turns=int(os.getenv("CONTEXT_KEEP_TURNS", "20")),
        project_files=os.getenv("CONTEXT_PROJECT_FILES", "1") in ("1", "true", "True"),
        repo_map=os.getenv("CONTEXT_REPO_MAP", "1") in ("1", "true", "True"),
        repo_map_max_lines=int(os.getenv("CONTEXT_REPO_MAP_MAX_LINES", "150")),
    )


class SecurityConfig(BaseModel):
    """安全边界参数（工作空间约束与 Bash 审批）。"""

    ask_policy: str = Field(default="ask", description="Bash 审批策略：ask/allow/deny")
    remember_choices: bool = Field(default=True, description="会话内是否记住审批选择，避免重复询问")


def load_security_config(env_file: str | Path = ".env") -> SecurityConfig:
    """从 .env 读取安全边界参数（与其余配置共用同一环境文件）。"""
    env_path = Path(env_file)
    if env_path.is_file():
        load_dotenv(env_path)
    else:
        load_dotenv()
    policy = os.getenv("SECURITY_ASK_POLICY", "ask").strip().lower()
    if policy not in ("ask", "allow", "deny"):
        policy = "ask"
    return SecurityConfig(
        ask_policy=policy,
        remember_choices=os.getenv("SECURITY_REMEMBER_CHOICES", "1") in ("1", "true", "True"),
    )


class RetentionConfig(BaseModel):
    """运行期数据保留参数（trace / transcript / artifact 的清理阈值）。"""

    keep_sessions: int = Field(default=30, description="保留最近多少个会话（超出裁剪）")
    max_age_days: int = Field(default=30, description="运行文件超过多少天视为过期并清理")


def load_retention_config(env_file: str | Path = ".env") -> RetentionConfig:
    """从 .env 读取数据保留参数（与其余配置共用同一环境文件）。"""
    env_path = Path(env_file)
    if env_path.is_file():
        load_dotenv(env_path)
    else:
        load_dotenv()
    return RetentionConfig(
        keep_sessions=int(os.getenv("RETENTION_KEEP_SESSIONS", "30")),
        max_age_days=int(os.getenv("RETENTION_MAX_AGE_DAYS", "30")),
    )
