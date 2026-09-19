"""Unit tests for dmd.config: role-based YAML config loading & env expansion."""

from __future__ import annotations

import pytest

from dmd.config import ConfigError, load_config, load_config_dict

# ---------------------------------------------------------------------------
# load_config_dict: happy path
# ---------------------------------------------------------------------------

def test_load_config_dict_happy_path_all_roles():
    """All five model roles validate and round-trip into typed AppConfig."""
    raw = {
        "models": {
            "synthesis": {
                "base_url": "http://localhost:8081/v1",
                "api_key": "literal-key",
                "model_id": "ornith-1.5-35b",
                "extra_body": {},
            },
            "fast": {
                "base_url": "http://localhost:8081/v1",
                "api_key": "literal-key",
                "model_id": "qwen3.8-4b",
                "extra_body": {"enable_thinking": False},
            },
            "vision": {
                "enabled": True,
                "base_url": "http://localhost:8081/v1",
                "api_key": "literal-key",
                "model_id": "vision-1",
            },
            "stt": {
                "base_url": "http://localhost:8081/v1",
                "api_key": "literal-key",
            },
            "embeddings": {
                "provider": "endpoint",
                "base_url": "http://localhost:8081/v1",
                "api_key": "literal-key",
                "model_id": "bge-small",
            },
        },
    }
    cfg = load_config_dict(raw)

    assert cfg.models.synthesis.model_id == "ornith-1.5-35b"
    assert cfg.models.synthesis.api_key == "literal-key"
    assert cfg.models.fast is not None
    assert cfg.models.fast.model_id == "qwen3.8-4b"
    assert cfg.models.fast.extra_body == {"enable_thinking": False}
    assert cfg.models.vision is not None
    assert cfg.models.vision.enabled is True
    assert cfg.models.vision.model_id == "vision-1"
    assert cfg.models.stt.base_url == "http://localhost:8081/v1"
    assert cfg.models.embeddings.provider == "endpoint"
    assert cfg.models.embeddings.model_id == "bge-small"


# ---------------------------------------------------------------------------
# env var expansion: success and failure
# ---------------------------------------------------------------------------

def test_env_var_expansion_success(monkeypatch):
    """${VAR} is substituted from the environment at load time."""
    monkeypatch.setenv("DMD_TEST_KEY", "expanded-secret-42")
    cfg = load_config_dict({
        "models": {
            "synthesis": {
                "base_url": "http://x",
                "model_id": "m",
                "api_key": "${DMD_TEST_KEY}",
            },
            "stt": {"base_url": "http://s"},
        }
    })
    assert cfg.models.synthesis.api_key == "expanded-secret-42"


def test_env_var_expansion_in_nested_extra_body(monkeypatch):
    """Expansion recurses into dicts/lists, including inside extra_body."""
    monkeypatch.setenv("DMD_NESTED", "from-env")
    cfg = load_config_dict({
        "models": {
            "synthesis": {
                "base_url": "http://x",
                "model_id": "m",
                "extra_body": {"tag": "${DMD_NESTED}", "list": ["${DMD_NESTED}", "plain"]},
            },
            "stt": {"base_url": "http://s"},
        }
    })
    assert cfg.models.synthesis.extra_body == {
        "tag": "from-env",
        "list": ["from-env", "plain"],
    }


def test_env_var_expansion_missing_raises(monkeypatch):
    """A ${VAR} reference with no env value surfaces as ConfigError."""
    monkeypatch.delenv("DMD_DEFINITELY_NOT_SET", raising=False)
    with pytest.raises(ConfigError, match="DMD_DEFINITELY_NOT_SET"):
        load_config_dict({
            "models": {
                "synthesis": {
                    "base_url": "http://x",
                    "model_id": "m",
                    "api_key": "${DMD_DEFINITELY_NOT_SET}",
                },
                "stt": {"base_url": "http://s"},
            }
        })


# ---------------------------------------------------------------------------
# missing synthesis role
# ---------------------------------------------------------------------------

def test_missing_synthesis_role_raises():
    """Synthesis is the only required generation role; omitting it errors."""
    with pytest.raises(ConfigError):
        load_config_dict({
            "models": {
                "stt": {"base_url": "http://s"},
            }
        })


# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------

def test_defaults_orchestration_max_concurrent_is_three():
    """When orchestration section is omitted, max_concurrent defaults to 3."""
    cfg = load_config_dict({
        "models": {
            "synthesis": {"base_url": "http://x", "model_id": "m"},
            "stt": {"base_url": "http://s"},
        }
    })
    assert cfg.orchestration.max_concurrent == 3
    # Other OrchestrationConfig defaults should also be applied.
    # job_timeout_s must exceed agent.agent_timeout_s (45.0) or the pool
    # silently kills every card (2026-09-05 live defect).
    assert cfg.orchestration.job_timeout_s == 60.0
    assert cfg.orchestration.stale_after_s == 120.0


def test_defaults_embeddings_provider_local():
    """When embeddings section is omitted, provider defaults to 'local'."""
    cfg = load_config_dict({
        "models": {
            "synthesis": {"base_url": "http://x", "model_id": "m"},
            "stt": {"base_url": "http://s"},
        }
    })
    assert cfg.models.embeddings.provider == "local"
    # And the default embedding model is populated.
    assert cfg.models.embeddings.model_id == "BAAI/bge-small-en-v1.5"


def test_defaults_stt_pipeline_and_project():
    """Stt pipeline and project blocks default to their built-in values."""
    cfg = load_config_dict({
        "models": {
            "synthesis": {"base_url": "http://x", "model_id": "m"},
            "stt": {"base_url": "http://s"},
        }
    })
    assert cfg.stt_pipeline.sample_rate == 16000
    # VAD endpoint hangover default (tuned low for the 5s voice->transcript
    # budget; see dmd/config.py SttPipelineConfig).
    assert cfg.stt_pipeline.silence_ms == 500
    assert cfg.stt_pipeline.min_utterance_ms == 400
    assert cfg.project.name == "Untitled Campaign"


# ---------------------------------------------------------------------------
# extra_body passthrough
# ---------------------------------------------------------------------------

def test_extra_body_passthrough_preserved():
    """extra_body dict (with nested structure) is preserved verbatim through validation."""
    payload = {
        "enable_thinking": False,
        "temperature": 0.7,
        "top_p": 0.9,
        "stop": ["</s>", "###"],
        "nested": {"key": "value", "deep": {"x": 1}},
    }
    cfg = load_config_dict({
        "models": {
            "synthesis": {
                "base_url": "http://x",
                "model_id": "m",
                "extra_body": payload,
            },
            "stt": {"base_url": "http://s", "extra_body": {"lang": "en"}},
        }
    })
    assert cfg.models.synthesis.extra_body == payload
    assert cfg.models.stt.extra_body == {"lang": "en"}


def test_extra_body_defaults_to_empty_dict_when_omitted():
    """If extra_body is not provided, it defaults to {} (not missing/None)."""
    cfg = load_config_dict({
        "models": {
            "synthesis": {"base_url": "http://x", "model_id": "m"},
            "stt": {"base_url": "http://s"},
        }
    })
    assert cfg.models.synthesis.extra_body == {}
    assert cfg.models.stt.extra_body == {}


# ---------------------------------------------------------------------------
# invalid yaml shape (load_config, file path)
# ---------------------------------------------------------------------------

def test_load_config_invalid_yaml_shape_list_raises(tmp_path):
    """A YAML file whose top-level is a list (not a mapping) raises ConfigError."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="YAML mapping"):
        load_config(str(cfg_file))


def test_load_config_invalid_yaml_shape_string_raises(tmp_path):
    """A YAML file whose top-level is a bare string raises ConfigError."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("just a string\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="YAML mapping"):
        load_config(str(cfg_file))


def test_load_config_dict_missing_synthesis_via_file_raises(tmp_path):
    """Pydantic ValidationError on missing synthesis is wrapped as ConfigError when loading from file."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "models:\n"
        "  stt:\n"
        "    base_url: http://s\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_config(str(cfg_file))


# ---------------------------------------------------------------------------
# api_key None tolerated
# ---------------------------------------------------------------------------

def test_api_key_none_explicit_tolerated():
    """api_key: None is accepted (Optional[str]) and preserved."""
    cfg = load_config_dict({
        "models": {
            "synthesis": {
                "base_url": "http://x",
                "model_id": "m",
                "api_key": None,
            },
            "stt": {"base_url": "http://s", "api_key": None},
        }
    })
    assert cfg.models.synthesis.api_key is None
    assert cfg.models.stt.api_key is None


def test_api_key_omitted_defaults_to_none():
    """Omitting api_key entirely is equivalent to None."""
    cfg = load_config_dict({
        "models": {
            "synthesis": {"base_url": "http://x", "model_id": "m"},
            "stt": {"base_url": "http://s"},
        }
    })
    assert cfg.models.synthesis.api_key is None
    assert cfg.models.stt.api_key is None
