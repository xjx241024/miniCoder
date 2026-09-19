"""RAG 相关配置加载测试：默认值与 .env 覆盖（隔离真实环境与真实 .env）。"""

import pytest
from dotenv import load_dotenv as real_load_dotenv

from core.config import load_embedding_config, load_rag_config


@pytest.fixture
def isolated_env(monkeypatch):
    """隔离配置环境：清空相关变量，并禁止 load_dotenv 回退读取真实 .env。

    不隔离时会读到开发者本机的 .env（含 API Key），
    既导致测试不稳定，也可能把密钥打印进失败信息。
    """
    keys = (
        "EMBEDDING_PROVIDER", "EMBEDDING_MODEL_ID", "EMBEDDING_API_KEY",
        "EMBEDDING_BASE_URL", "EMBEDDING_TIMEOUT", "EMBEDDING_MAX_RETRIES",
        "EMBEDDING_RETRY_BACKOFF", "EMBEDDING_MAX_BATCH",
        "RAG_CHUNK_SIZE", "RAG_CHUNK_OVERLAP", "RAG_INDEX_MAX_CHARS",
    )
    for key in keys:
        monkeypatch.delenv(key, raising=False)

    def fake_load_dotenv(dotenv_path=None, **kwargs):
        # 显式传入的文件才加载，且强制覆盖进程内已有值，保证测试确定性
        if dotenv_path is None:
            return False
        return real_load_dotenv(dotenv_path, override=True, **kwargs)

    monkeypatch.setattr("core.config.load_dotenv", fake_load_dotenv)


def test_load_embedding_config_defaults(tmp_path, isolated_env):
    config = load_embedding_config(tmp_path / "missing.env")
    assert config.provider == "siliconflow"
    assert config.model_id == "BAAI/bge-m3"
    assert config.base_url == "https://api.siliconflow.cn/v1"
    assert config.api_key == ""
    assert config.max_batch == 32


def test_load_embedding_config_from_env_file(tmp_path, isolated_env):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "EMBEDDING_PROVIDER=openai\n"
        "EMBEDDING_MODEL_ID=text-embedding-3-small\n"
        "EMBEDDING_API_KEY=sk-test\n"
        "EMBEDDING_BASE_URL=https://api.openai.com/v1\n"
        "EMBEDDING_MAX_BATCH=8\n",
        encoding="utf-8",
    )
    config = load_embedding_config(env_file)
    assert config.provider == "openai"
    assert config.model_id == "text-embedding-3-small"
    assert config.api_key == "sk-test"
    assert config.base_url == "https://api.openai.com/v1"
    assert config.max_batch == 8


def test_load_rag_config_defaults_and_override(tmp_path, isolated_env):
    assert load_rag_config(tmp_path / "missing.env").chunk_size == 800
    env_file = tmp_path / ".env"
    env_file.write_text(
        "RAG_CHUNK_SIZE=500\nRAG_CHUNK_OVERLAP=50\nRAG_INDEX_MAX_CHARS=1000\n",
        encoding="utf-8",
    )
    config = load_rag_config(env_file)
    assert config.chunk_size == 500
    assert config.chunk_overlap == 50
    assert config.index_max_chars == 1000
