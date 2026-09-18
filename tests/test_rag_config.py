"""RAG 相关配置加载测试：默认值与 .env 覆盖。"""

from core.config import load_embedding_config, load_rag_config


def test_load_embedding_config_defaults(tmp_path):
    config = load_embedding_config(tmp_path / "missing.env")
    assert config.provider == "siliconflow"
    assert config.model_id == "BAAI/bge-m3"
    assert config.base_url == "https://api.siliconflow.cn/v1"
    assert config.api_key == ""
    assert config.max_batch == 32


def test_load_embedding_config_from_env_file(tmp_path):
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


def test_load_rag_config_defaults_and_override(tmp_path):
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
