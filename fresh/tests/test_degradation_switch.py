"""public.degradation swaps the version and cache directory, then restores them."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bench
import config
import data_io
import public


def test_switch_restores():
    before = config.DEGRADE, data_io.PUBLIC_CACHE
    with public.degradation(2):
        assert config.DEGRADE == 2
        assert data_io.PUBLIC_CACHE.name == "cache-v2"
    assert (config.DEGRADE, data_io.PUBLIC_CACHE) == before


def test_external_only_refit_selects_v4_decoder_search(monkeypatch):
    monkeypatch.setattr(config, "DEGRADE", 1)
    monkeypatch.delenv("FRESH_TRAIN_DEGRADE", raising=False)
    assert bench.dtag("setp-gbm", dur=True) == "setp-gbm-dur"
    monkeypatch.setenv("FRESH_TRAIN_DEGRADE", "4")
    assert bench.dtag("setp-gbm", dur=True) == "setp-gbm-dur-g4"
