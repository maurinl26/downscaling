"""Tests du modelstore (Phase 4) — sans réseau ni torch lourd.

Couvre le plan de ``fetch-pretrained`` (manifeste, filtres, idempotence du
``--list``) et la résolution d'artefacts du loader Prithvi : ``modelstore/``
d'abord, repli HuggingFace ensuite. ``hf_hub_download`` est monkeypatché — aucun
téléchargement réel.
"""

from __future__ import annotations

import pytest

from downscaling.scripts import fetch_pretrained as fp


# ---------------------------------------------------------------------------
# fetch_pretrained : plan / manifeste
# ---------------------------------------------------------------------------

def test_manifest_keys_unique_and_known():
    keys = [a.key for a in fp.MANIFEST]
    assert keys == sorted(set(keys), key=keys.index)  # pas de doublon
    assert {"backbone", "granite"} <= set(keys)


def test_plan_default_returns_all():
    assert fp.plan() == list(fp.MANIFEST)


def test_plan_only_filters():
    sel = fp.plan("granite")
    assert len(sel) == 1 and sel[0].key == "granite"


def test_plan_unknown_exits():
    with pytest.raises(SystemExit):
        fp.plan("nope")


def test_artifact_dest_under_root(tmp_path):
    art = fp.MANIFEST[0]
    assert art.dest(tmp_path) == tmp_path / art.local_dir


def test_fetch_one_calls_snapshot(monkeypatch, tmp_path):
    """fetch_one délègue à snapshot_download avec le bon repo/dest/patterns."""
    huggingface_hub = pytest.importorskip("huggingface_hub")
    calls = {}

    def fake_snapshot(repo_id, local_dir, allow_patterns):
        calls.update(repo_id=repo_id, local_dir=local_dir, allow_patterns=allow_patterns)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)

    art = fp.plan("granite")[0]
    dest = fp.fetch_one(art, tmp_path)
    assert dest == tmp_path / "granite-downscaling"
    assert dest.is_dir()  # créé même avant download
    assert calls["repo_id"] == art.repo_id
    assert calls["allow_patterns"] == art.patterns


# ---------------------------------------------------------------------------
# loader : résolution modelstore d'abord, HF en repli
# ---------------------------------------------------------------------------

def test_resolve_artifact_prefers_modelstore(monkeypatch, tmp_path):
    loader = pytest.importorskip("downscaling.prtihvi_wxc.loader")

    # modelstore/ peuplé : on doit retourner le chemin local, sans appel HF.
    monkeypatch.setattr(loader, "MODELSTORE", tmp_path)
    sub = tmp_path / "granite-downscaling"
    sub.mkdir(parents=True)
    (sub / "model.safetensors").write_bytes(b"")

    def fail_download(**kwargs):
        raise AssertionError("hf_hub_download ne doit pas être appelé si présent localement")

    monkeypatch.setattr(loader, "hf_hub_download", fail_download)

    got = loader.resolve_artifact(loader.GRANITE_DOWNSCALING_REPO, "model.safetensors")
    assert got == str(sub / "model.safetensors")


def test_resolve_artifact_falls_back_to_hf(monkeypatch, tmp_path):
    loader = pytest.importorskip("downscaling.prtihvi_wxc.loader")

    monkeypatch.setattr(loader, "MODELSTORE", tmp_path)  # vide → repli HF

    def fake_download(repo_id, filename):
        return f"/hf-cache/{repo_id}/{filename}"

    monkeypatch.setattr(loader, "hf_hub_download", fake_download)

    got = loader.resolve_artifact(loader.GRANITE_DOWNSCALING_REPO, "model.safetensors")
    assert got == f"/hf-cache/{loader.GRANITE_DOWNSCALING_REPO}/model.safetensors"


def test_resolve_device_auto(monkeypatch):
    loader = pytest.importorskip("downscaling.prtihvi_wxc.loader")
    torch = loader.torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert loader.resolve_device("auto") == "cpu"
    assert loader.resolve_device("cuda:0") == "cuda:0"  # explicite inchangé

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert loader.resolve_device("auto") == "cuda"
