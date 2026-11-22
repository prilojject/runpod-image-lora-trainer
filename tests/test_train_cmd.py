from src.config import resolve_config
from src.train import build_command, build_prompts, write_dataset_toml, write_prompt_file


def test_sdxl_command(tmp_path):
    cfg = resolve_config({"dataset_url": "https://x/y.zip", "trigger_word": "zxowl", "steps": 30}, env={})
    p = tmp_path / "p.txt"
    write_prompt_file(cfg, build_prompts(cfg), str(p))
    assert "--s 28" in p.read_text() and "zxowl" in p.read_text()
    cmd = build_command(cfg, {"checkpoint": "/m/sdxl.safetensors"}, "/d.toml", "/o", "n", str(p), 24.0)
    s = " ".join(cmd)
    assert "sdxl_train_network.py" in s and "--network_train_unet_only" in s
    assert "--max_train_steps=30" in s and "--sample_every_n_steps=30" in s


def test_flux_command():
    cfg = resolve_config({"dataset_url": "https://x/y.zip", "base_model": "flux-dev", "steps": 50},
                         env={"HF_TOKEN": "hf_abcdefgh"})
    models = {"checkpoint": "/f", "ae": "/a", "clip_l": "/c", "t5xxl": "/t"}
    s = " ".join(build_command(cfg, models, "/d.toml", "/o", "n", None, 48.0))
    assert "flux_train_network.py" in s and "networks.lora_flux" in s and "--fp8_base" in s
    assert "--blocks_to_swap" not in s
    s24 = " ".join(build_command(cfg, models, "/d.toml", "/o", "n", None, 24.0))
    assert "--blocks_to_swap=8" in s24


def test_dataset_toml(tmp_path):
    cfg = resolve_config({"dataset_url": "https://x/y.zip", "preset": "style"}, env={})
    t = tmp_path / "d.toml"
    write_dataset_toml(cfg, "/imgs", str(t), 6)
    txt = t.read_text()
    assert 'image_dir = "/imgs"' in txt and "flip_aug = true" in txt and "num_repeats = 16" in txt
