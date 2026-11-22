import pytest

from src.config import ConfigError, render_prompt, resolve_config

URL = "https://example.com/data.zip"


def test_defaults_and_preset():
    c = resolve_config({"dataset_url": URL, "trigger_word": "zxowl"}, env={})
    assert c.base_model == "sdxl" and c.preset == "object"
    assert c.steps == 1200 and c.rank == 16 and c.alpha == 16.0
    assert c.sources["steps"] == "preset"
    assert len(c.sample_prompts) == 2


def test_input_beats_env_beats_preset():
    c = resolve_config({"dataset_url": URL, "steps": 40}, env={"STEPS": "500", "LORA_RANK": "8", "PRESET": "style"})
    assert c.steps == 40 and c.rank == 8 and c.preset == "style"
    assert c.flip_aug is True  # from the style preset


def test_tokens_rejected():
    with pytest.raises(ConfigError, match="tokens"):
        resolve_config({"dataset_url": URL, "hf_token": "hf_x"}, env={})


def test_flux_needs_token():
    with pytest.raises(ConfigError, match="gated"):
        resolve_config({"dataset_url": URL, "base_model": "flux-dev"}, env={})
    c = resolve_config({"dataset_url": URL, "base_model": "flux-dev"}, env={"HF_TOKEN": "hf_abcdefgh"})
    assert c.is_flux and any("Non-Commercial" in w for w in c.warnings)


def test_dataset_required_and_exclusive():
    with pytest.raises(ConfigError):
        resolve_config({}, env={})
    with pytest.raises(ConfigError):
        resolve_config({"dataset_url": URL, "hf_dataset": "a/b"}, env={})
    c = resolve_config({"hf_dataset": "a/b"}, env={"DATASET_URL": URL})
    assert c.dataset_url == "" and c.hf_dataset == "a/b"


def test_validation_errors():
    with pytest.raises(ConfigError) as e:
        resolve_config({"dataset_url": "ftp://x", "steps": 5, "resolution": 1000, "bogus": 1}, env={})
    msg = str(e.value)
    for frag in ("dataset_url", "steps", "resolution", "unknown"):
        assert frag in msg


def test_render_prompt():
    assert render_prompt("a photo of {trigger} on a table", "zxowl", "figurine") == "a photo of zxowl figurine on a table"
    assert render_prompt("{trigger} style", "", "") == "the subject style"
