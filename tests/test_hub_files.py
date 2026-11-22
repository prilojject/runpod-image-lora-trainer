"""Static checks of .runpod/hub.json and tests.json against the Runpod Hub publishing guide."""
import json
import os

from src.config import PARAMS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRA_ENV = {"HF_TOKEN", "TRAIN_TIMEOUT_SECONDS"}


def load(name):
    with open(os.path.join(ROOT, ".runpod", name)) as fh:
        return json.load(fh)


def test_required_files():
    for f in ("handler.py", "Dockerfile", "README.md", "LICENSE", ".runpod/hub.json", ".runpod/tests.json",
              ".runpod/icon.png"):
        assert os.path.exists(os.path.join(ROOT, f)), f


def test_hub_json():
    hub = load("hub.json")
    for k in ("title", "description", "type", "category", "config", "iconUrl"):
        assert hub.get(k), k
    assert hub["type"] == "serverless" and hub["category"] == "image"
    cfg = hub["config"]
    assert cfg["runsOn"] == "GPU" and isinstance(cfg["containerDiskInGb"], int) and cfg["gpuCount"] == 1
    known = {p.env for p in PARAMS if p.env} | EXTRA_ENV
    keys = [e["key"] for e in cfg["env"]]
    assert len(keys) == len(set(keys)) and set(keys) <= known, set(keys) - known
    for preset in cfg["presets"]:
        assert set(preset["defaults"]) <= set(keys), preset["name"]
    for e in cfg["env"]:
        inp = e["input"]
        assert inp["type"] in {"string", "number", "boolean", "huggingface"}
        if inp["type"] == "number":
            assert inp["min"] <= inp["default"] <= inp["max"]
        if inp["type"] == "boolean":
            assert {inp["trueValue"], inp["falseValue"]} == {"true", "false"}
        for o in inp.get("options", []):
            assert o["label"] and o["value"]


def test_tests_json():
    t = load("tests.json")
    assert t["config"]["gpuTypeId"] and t["config"]["gpuCount"] == 1
    for case in t["tests"]:
        assert case["name"] and case["input"]
        assert case["input"]["steps"] <= 60  # must fit the Hub test timeout
        assert case.get("timeout", 0) >= 600000
