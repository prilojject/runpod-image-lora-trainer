import os
import zipfile

from PIL import Image

from src.config import resolve_config
from src.dataset import _collect_pairs, _extract, _write_caption


def _make_zip(tmp_path, n=3, with_caps=True):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(n):
        Image.new("RGB", (512, 512), (i * 40, 100, 100)).save(src / f"img{i}.jpg")
        if with_caps and i % 2 == 0:
            (src / f"img{i}.txt").write_text(f"caption {i}")
    (src / ".DS_Store").write_text("x")
    zp = tmp_path / "d.zip"
    with zipfile.ZipFile(zp, "w") as z:
        for f in os.listdir(src):
            z.write(src / f, f"folder/{f}")
        z.writestr("../evil.txt", "nope")
    return zp


def test_extract_and_pairs(tmp_path):
    zp = _make_zip(tmp_path)
    out = tmp_path / "out"
    _extract(str(zp), str(out))
    assert not (tmp_path / "evil.txt").exists()  # zip-slip blocked
    pairs = _collect_pairs(str(out))
    assert len(pairs) == 3
    assert pairs[0][1] == "caption 0" and pairs[1][1] is None


def test_trigger_prepended(tmp_path):
    cfg = resolve_config({"dataset_url": "https://x/y.zip", "trigger_word": "zxowl"}, env={})
    img = tmp_path / "a.jpg"
    _write_caption(str(img), "a photo on a table", cfg)
    assert (tmp_path / "a.txt").read_text().strip() == "zxowl, a photo on a table"
    _write_caption(str(img), "a photo of zxowl", cfg)
    assert (tmp_path / "a.txt").read_text().strip() == "a photo of zxowl"
