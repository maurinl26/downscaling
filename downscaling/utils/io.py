"""Output IO helpers: write zarr + sidecar files to local path or S3 (Scaleway).

Convention: scripts accept ``--out`` as either a local directory or an
``s3://bucket/prefix`` / ``scw://bucket/prefix`` URL. The helpers below
pick the right backend (local Path vs fsspec mapper) so the caller does
not have to branch.

Scaleway endpoint resolution order:
    1. ``$AWS_ENDPOINT_URL``
    2. ``$SCW_S3_ENDPOINT_URL``
    3. ``https://s3.fr-par.scw.cloud`` (default)

Credentials are read from the standard boto3/fsspec environment
(``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``) or ``~/.aws/credentials``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Union

SCW_DEFAULT_ENDPOINT = "https://s3.fr-par.scw.cloud"


def is_remote(path: Union[str, Path]) -> bool:
    """True if the path is an ``s3://`` or ``scw://`` URL."""
    s = str(path)
    return s.startswith(("s3://", "scw://"))


def _normalize_s3_url(url: str) -> str:
    """Normalize ``scw://`` to ``s3://`` (we use the S3 protocol everywhere)."""
    if url.startswith("scw://"):
        return "s3://" + url[len("scw://") :]
    return url


def _s3_endpoint() -> str:
    return (
        os.environ.get("AWS_ENDPOINT_URL")
        or os.environ.get("SCW_S3_ENDPOINT_URL")
        or SCW_DEFAULT_ENDPOINT
    )


def _fs():
    import fsspec

    return fsspec.filesystem("s3", client_kwargs={"endpoint_url": _s3_endpoint()})


def make_zarr_store(out: Union[str, Path], year: int) -> Any:
    """Return a target for ``xarray.Dataset.to_zarr`` / ``xr.open_zarr``.

    - Local path → ``str`` path to ``<out>/<year>.zarr`` (parent dir created).
    - ``s3://bucket/key`` or ``scw://bucket/key`` → fsspec mapper for
      ``<key>/<year>.zarr`` with the Scaleway endpoint configured.
    """
    out_s = str(out)
    if is_remote(out_s):
        url = _normalize_s3_url(out_s).rstrip("/")
        key = f"{url[len('s3://') :]}/{year}.zarr"
        return _fs().get_mapper(key)
    out_path = Path(out_s)
    out_path.mkdir(parents=True, exist_ok=True)
    return str(out_path / f"{year}.zarr")


def write_sidecar(out: Union[str, Path], year: int, suffix: str, content: str) -> str:
    """Write a sidecar text file next to the year zarr.

    ``suffix`` should include the dot, e.g. ``.metadata.json``.
    Returns the full URL or path where it was written.
    """
    out_s = str(out)
    if is_remote(out_s):
        import fsspec

        url = _normalize_s3_url(out_s).rstrip("/")
        full = f"{url}/{year}{suffix}"
        with fsspec.open(
            full,
            mode="w",
            client_kwargs={"endpoint_url": _s3_endpoint()},
        ) as f:
            f.write(content)
        return full
    out_path = Path(out_s)
    out_path.mkdir(parents=True, exist_ok=True)
    target = out_path / f"{year}{suffix}"
    target.write_text(content)
    return str(target)


def describe(out: Union[str, Path], year: int, suffix: str = ".zarr") -> str:
    """Return a human-friendly path string for logging (no IO)."""
    out_s = str(out)
    if is_remote(out_s):
        return f"{_normalize_s3_url(out_s).rstrip('/')}/{year}{suffix}"
    return str(Path(out_s) / f"{year}{suffix}")
