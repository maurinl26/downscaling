"""Tests pour les helpers root local/S3 de ``analyze_recalibrated_statistical``.

On teste le routage local vs s3:// (pas l'accès S3 réel — couvert par les
smoke tests d'intégration). Les helpers sont l'apport de l'issue #55.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from downscaling.scripts.analyze_recalibrated_statistical import (
    _is_remote,
    _list_zarrs,
    _storage_options,
    _zarr_stem,
)


class TestIsRemote:
    def test_local_path_str_is_not_remote(self):
        assert not _is_remote("/tmp/output/zarr")
        assert not _is_remote("./relative/path")
        assert not _is_remote("file:///etc")

    def test_s3_uri_is_remote(self):
        assert _is_remote("s3://bucket/key")

    def test_gs_uri_is_remote(self):
        assert _is_remote("gs://bucket/key")

    def test_empty_string_is_not_remote(self):
        assert not _is_remote("")


class TestStorageOptions:
    def test_no_endpoint_env_returns_skip_cache_only(self, monkeypatch):
        monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
        monkeypatch.delenv("AWS_S3_ENDPOINT", raising=False)
        # skip_instance_cache=True est toujours présent (fraîcheur s3fs en boucle LOO).
        assert _storage_options() == {"skip_instance_cache": True}

    def test_aws_endpoint_url_wins(self, monkeypatch):
        monkeypatch.setenv("AWS_ENDPOINT_URL", "https://s3.fr-par.scw.cloud")
        monkeypatch.delenv("AWS_S3_ENDPOINT", raising=False)
        opts = _storage_options()
        assert opts == {
            "skip_instance_cache": True,
            "client_kwargs": {"endpoint_url": "https://s3.fr-par.scw.cloud"},
        }

    def test_legacy_aws_s3_endpoint_fallback(self, monkeypatch):
        monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
        monkeypatch.setenv("AWS_S3_ENDPOINT", "https://s3.legacy.cloud")
        opts = _storage_options()
        assert opts == {
            "skip_instance_cache": True,
            "client_kwargs": {"endpoint_url": "https://s3.legacy.cloud"},
        }

    def test_aws_endpoint_url_preferred_over_legacy(self, monkeypatch):
        monkeypatch.setenv("AWS_ENDPOINT_URL", "https://canonical")
        monkeypatch.setenv("AWS_S3_ENDPOINT", "https://legacy")
        opts = _storage_options()
        assert opts["client_kwargs"]["endpoint_url"] == "https://canonical"


class TestZarrStem:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("2022.zarr", "2022"),
            ("/local/path/2023.zarr", "2023"),
            ("/local/path/2023.zarr/", "2023"),
            ("s3://bucket/prefix/2024.zarr", "2024"),
            ("s3://bucket/prefix/2024.zarr/", "2024"),
        ],
    )
    def test_extracts_year_stem(self, url, expected):
        assert _zarr_stem(url) == expected


class TestListZarrsLocal:
    def test_lists_zarr_dirs_local(self, tmp_path: Path):
        # Create fake zarr "dirs" (just empty dirs with .zarr suffix —
        # _list_zarrs uses Path.glob which matches by name, not validity).
        (tmp_path / "2022.zarr").mkdir()
        (tmp_path / "2023.zarr").mkdir()
        (tmp_path / "not-a-zarr.txt").write_text("ignore me")
        result = _list_zarrs(str(tmp_path))
        assert len(result) == 2
        assert all(r.endswith(".zarr") for r in result)
        # Sorted ascending
        assert _zarr_stem(result[0]) == "2022"
        assert _zarr_stem(result[1]) == "2023"

    def test_empty_dir_returns_empty_list(self, tmp_path: Path):
        assert _list_zarrs(str(tmp_path)) == []

    def test_returns_string_paths_not_path_objects(self, tmp_path: Path):
        (tmp_path / "2022.zarr").mkdir()
        result = _list_zarrs(str(tmp_path))
        assert all(isinstance(r, str) for r in result)
