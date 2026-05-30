"""Tests de composition Hydra (``configs/``).

Vérifie que la config racine et chaque ``experiment=`` composent sans erreur et
exposent les clés attendues par l'entry point ``downscaling-run``. C'est le
filet qui empêche un override de région de casser la chaîne silencieusement.
"""

from __future__ import annotations

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from downscaling.paths import CONFIG_DIR


def _compose(overrides=None):
    # config_dir absolu : indispensable depuis un entry point console (cf. paths.py).
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name="config", overrides=overrides or [])


def test_config_dir_exists():
    assert CONFIG_DIR.is_dir(), f"{CONFIG_DIR} introuvable"
    assert (CONFIG_DIR / "config.yaml").is_file()


def test_default_config_composes():
    cfg = _compose()
    # Les groupes attendus sont présents.
    for group in ("domain", "data", "statistical", "dl", "indices", "cluster", "run"):
        assert group in cfg, f"groupe '{group}' manquant"


def test_drome_ardeche_values():
    cfg = _compose(["experiment=drome_ardeche"])
    assert cfg.domain.nx == 118
    assert cfg.domain.ny == 167
    # Gradient mensuel : 12 valeurs, toutes négatives (refroidissement en altitude).
    gamma = OmegaConf.to_container(cfg.statistical.lapse_rate.monthly_gamma)
    assert len(gamma) == 12
    assert all(g < 0 for g in gamma)
    # Le run par défaut de l'expérience calcule les indices.
    assert cfg.run.compute_indices is True
    assert cfg.run.date == "20210427"


def test_cluster_override():
    local = _compose(["cluster=local"])
    cloud = _compose(["cluster=cloud"])
    assert local.cluster.precision == "32-true"      # MPS-safe
    assert cloud.cluster.accelerator == "gpu"


def test_dotlist_override_reaches_leaf():
    cfg = _compose(["statistical.quantile_mapping.enabled=false"])
    assert cfg.statistical.quantile_mapping.enabled is False
    # Inchangé par ailleurs.
    assert cfg.statistical.quantile_mapping.n_quantiles == 100
