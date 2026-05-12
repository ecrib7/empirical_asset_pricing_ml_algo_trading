"""
tests/test_outputs_contract.py
------------------------------
Dashboard contract tests over real, training-derived outputs in
``outputs/<variant>/``.

For every variant that ships a ``metrics.json`` (i.e. is exposed in
the dashboard variant selector) we assert that the on-disk artifacts
match the schema the dashboard expects to read. These tests do NOT
regenerate or mutate any output; they only inspect what training has
already produced.

Contract checked per variant:

  * Required files exist: metrics.json, comprehensive.csv, oos_r2.csv,
    regimes.csv, portfolio_returns.pkl.
  * metrics.json has the ``_ensembles`` block with ENS-AVG/ENS-MSE
    constituents, and ENS-MSE weights sum to 1.
  * comprehensive.csv exposes the canonical columns including ``SR*``
    and ``Max DD (%)``; ``SR*`` is not entirely NaN where defined.
  * regimes.csv carries the full per-regime schema.
  * portfolio_returns.pkl is the bundle_v1 dict (net/gross/turnover)
    and every model has H-L Series in each section.
  * Model names are consistent across metrics.json, comprehensive.csv
    and portfolio_returns.pkl.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUTPUTS_DIR = ROOT / "outputs"

REQUIRED_FILES = (
    "metrics.json",
    "comprehensive.csv",
    "oos_r2.csv",
    "regimes.csv",
    "portfolio_returns.pkl",
)

COMPREHENSIVE_REQUIRED_COLS = {
    "Model",
    "Sharpe (net)",
    "Sharpe (gross)",
    "SR*",
    "Max DD (%)",
    "OOS R² (%)",
    "Mean TO (1-way)",
    "Alpha (% / yr)",
    "t(alpha)",
}

REGIMES_REQUIRED_COLS = {
    "regime_kind",
    "regime",
    "model",
    "sharpe_net",
    "sharpe_gross",
    "mean_return",
    "max_dd_pct",
    "skew",
    "kurt",
    "n_months",
}


def _discover_variants() -> list[str]:
    if not OUTPUTS_DIR.is_dir():
        return []
    return sorted(
        d.name
        for d in OUTPUTS_DIR.iterdir()
        if d.is_dir() and (d / "metrics.json").exists()
    )


VARIANTS = _discover_variants()


@pytest.fixture(scope="module")
def variant_path(request) -> Path:
    return OUTPUTS_DIR / request.param


pytestmark = pytest.mark.skipif(
    not VARIANTS, reason="no outputs/<variant>/metrics.json directories on disk"
)


@pytest.mark.parametrize("variant", VARIANTS)
class TestVariantOutputsContract:
    """One parametrised class per discovered variant — failure pinpoints which one."""

    def test_required_files_exist(self, variant):
        vdir = OUTPUTS_DIR / variant
        for name in REQUIRED_FILES:
            assert (vdir / name).exists(), f"{variant}: missing {name}"

    def test_metrics_json_has_ensembles_block(self, variant):
        vdir = OUTPUTS_DIR / variant
        metrics = json.loads((vdir / "metrics.json").read_text())
        assert "_ensembles" in metrics, f"{variant}: metrics.json missing _ensembles"
        ens = metrics["_ensembles"]
        assert "ENS-AVG" in ens, f"{variant}: ENS-AVG missing from _ensembles"
        assert "ENS-MSE" in ens, f"{variant}: ENS-MSE missing from _ensembles"
        for key in ("ENS-AVG", "ENS-MSE"):
            constituents = ens[key].get("constituents")
            assert constituents, f"{variant}: {key} has no constituents"
            assert all(isinstance(c, str) for c in constituents)

    def test_ens_mse_weights_sum_to_one(self, variant):
        vdir = OUTPUTS_DIR / variant
        metrics = json.loads((vdir / "metrics.json").read_text())
        weights = metrics.get("_ensembles", {}).get("ENS-MSE", {}).get("weights")
        assert weights, f"{variant}: ENS-MSE weights missing"
        total = sum(weights.values())
        assert total == pytest.approx(1.0, abs=1e-6), (
            f"{variant}: ENS-MSE weights sum {total} != 1.0"
        )

    def test_comprehensive_schema(self, variant):
        vdir = OUTPUTS_DIR / variant
        comp = pd.read_csv(vdir / "comprehensive.csv")
        missing = COMPREHENSIVE_REQUIRED_COLS - set(comp.columns)
        assert not missing, f"{variant}: comprehensive.csv missing {missing}"
        # SR* must have at least one finite value (some rows can be NaN if
        # the t-stat is undefined, but the whole column being NaN means the
        # dashboard would render an empty risk-adjusted Sharpe everywhere).
        assert comp["SR*"].notna().any(), f"{variant}: SR* is entirely NaN"

    def test_regimes_schema(self, variant):
        vdir = OUTPUTS_DIR / variant
        rg = pd.read_csv(vdir / "regimes.csv")
        missing = REGIMES_REQUIRED_COLS - set(rg.columns)
        assert not missing, f"{variant}: regimes.csv missing {missing}"
        assert len(rg) > 0, f"{variant}: regimes.csv is empty"

    def test_portfolio_bundle_format(self, variant):
        vdir = OUTPUTS_DIR / variant
        with open(vdir / "portfolio_returns.pkl", "rb") as f:
            bundle = pickle.load(f)
        assert isinstance(bundle, dict)
        assert bundle.get("_format") == "bundle_v1", (
            f"{variant}: portfolio bundle is not bundle_v1"
        )
        for section in ("net", "gross", "turnover"):
            assert section in bundle, f"{variant}: missing {section} section"
            assert bundle[section], f"{variant}: {section} section is empty"
            # H-L Series must exist for every model in each section
            for model, deciles in bundle[section].items():
                assert "H-L" in deciles, (
                    f"{variant}: model {model} missing H-L in {section}"
                )
                hl = deciles["H-L"]
                assert isinstance(hl, pd.Series), (
                    f"{variant}: {section}/{model}/H-L is not a Series"
                )
                assert len(hl) > 0, f"{variant}: {section}/{model}/H-L empty"

    def test_model_names_consistent_across_artifacts(self, variant):
        vdir = OUTPUTS_DIR / variant
        metrics = json.loads((vdir / "metrics.json").read_text())
        metric_models = {k for k in metrics if not k.startswith("_")}

        comp = pd.read_csv(vdir / "comprehensive.csv")
        comp_models = set(comp["Model"].unique().tolist())

        with open(vdir / "portfolio_returns.pkl", "rb") as f:
            bundle = pickle.load(f)
        pkl_models = set(bundle["net"].keys())

        assert metric_models == comp_models == pkl_models, (
            f"{variant}: model names diverge — "
            f"metrics={sorted(metric_models)} "
            f"comp={sorted(comp_models)} "
            f"pkl={sorted(pkl_models)}"
        )

    def test_ensemble_models_present(self, variant):
        vdir = OUTPUTS_DIR / variant
        comp = pd.read_csv(vdir / "comprehensive.csv")
        models = set(comp["Model"].unique().tolist())
        # Both ensembles must surface in the comprehensive table — the
        # dashboard's "Forecast Combination" tab expects them.
        for ens in ("ENS-AVG", "ENS-MSE"):
            assert ens in models, f"{variant}: {ens} missing from comprehensive.csv"
