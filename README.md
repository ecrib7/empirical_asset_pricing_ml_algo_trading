# GKX (2019) — Empirical Asset Pricing via Machine Learning
### IEOR 4733: Algorithmic Trading — Course Project
**Authors:** Francesco Pio De Girolamo · Brice Namy

This repository implements a **reproduction and extension** of **Gu, Kelly & Xiu (2020)** (*Empirical Asset Pricing via Machine Learning*, *Review of Financial Studies*): monthly panels, machine-learning return forecasts, recursive out-of-sample evaluation, long-short decile portfolios, and economic significance metrics.

---

## For the grader

There are two ways to interact with this project.

### Option A — Explore the results in the local dashboard (no training required)

You can access the results directly at https://empiricalassetpricingmlalgotrading-ohi72rrh526mb2iytyecgx.streamlit.app/

You can also run the dashboard locally on your computer:
The `outputs/paper/` and `outputs/improved/` directories already contain the artifacts the dashboard reads. Just install dependencies and launch:

```bash
pip install -r requirements.txt
streamlit run src/dashboard/app.py
```

The dashboard opens in your browser with a variant selector (`paper` / `improved`) in the sidebar. Tabs cover OOS R², comprehensive metrics, Diebold-Mariano heatmaps, portfolio returns, transaction-cost sensitivity, forecast-combination ensembles, regime-conditional evaluation, variable importance, and paper-vs-improved comparison.

### Option B — Reproduce all results from scratch in Google Colab

1. Zip this entire project as `empirical_asset_pricing_ml.zip`.
2. In Google Drive, create a folder named **exactly** `Algorithmic Trading Project` and upload the zip there.
3. Open `notebooks/empirical_asset_pricing_ml.ipynb` in Google Colab.
5. Run all cells top to bottom. Section 0 mounts Drive and unzips the project; Sections 3–5 run the WRDS data pipeline, train all 12 models, and evaluate them; Section 6 launches the dashboard inside Colab via ngrok.
6. End-to-end runtime is ≈ 24 hrs per variant on a Colab Pro T4. Cell-by-cell, Section 7 runs both `paper` and `improved` back-to-back.

WRDS access is required for the data fetch step (your own credentials, not provided in this repo).

---

## Two pipelines, one CLI flag

| Variant     | Sample period | Macro × char | Industry dummies | Transaction costs                | Forecast combination | Regime analysis |
|-------------|---------------|:-:|:-:|---|:-:|:-:|
| `paper`     | 1957 – 2016   | ✅ | ✅ | **0 bps** (gross, matches paper headline) | ✅ | ✅ |
| `improved`  | 1957 – 2024   | ✅ | ✅ | **Impact-aware** (FIM-style)              | ✅ | ✅ |

Each variant writes to its own `outputs/<variant>/` directory and uses its own cached feature matrix at `data/cache/feature_matrix_<variant>.parquet` — they don't overwrite each other.

### Sample-period details

* **1957 – 2016** — paper reproduction window.
* **2017 – 2024** — extension to test whether GKX-vintage signals survive post-publication. Includes COVID shock (Feb–Apr 2020) and the 2022–23 rate-hike cycle. Adds ~96 months of out-of-sample data (~30 % more than `paper`).

## What's in the improved pipeline

**Impact-aware transaction costs** (Frazzini-Israel-Moskowitz 2018-style). Per-stock per-month cost rate:

```
cost_bps_i = half_spread_bps(log_mcap_i) + λ × √(trade$ / ADV_i)
```

The half-spread is log-linearly interpolated between 25 bps for the smallest-cap decile and 5 bps for the largest, computed against each month's cross-sectional distribution of log market equity. The impact term scales with √(trade dollar / average daily $-volume). ADV is computed as monthly $-volume / 21 (trading days). Implementation: `src/backtest/engine.py::ImpactAwareTransactionCostModel`. The `paper` variant keeps the flat 0-bps cost so the headline Sharpe matches the paper's gross numbers.

**Forecast combination.** After per-model training, `--mode evaluate` automatically constructs two ensembles:

* `ENS-AVG` — equal-weighted average of all per-model predictions
* `ENS-MSE` — weighted average with weights ∝ 1 / validation MSE (validation slice = earliest 10 % of test dates)

Each ensemble is then routed through the same decile portfolio construction and gets its own row in the comprehensive table, DM matrix, and the rest of the metrics. Skip with `--no-ensembles`.

**Regime-conditional evaluation.** `--mode regimes` slices each model's H-L return series by:

* NBER recession vs expansion (real recession dates: 1990-91, 2001, 2007-09, 2020)
* VIX terciles (low / mid / high implied vol — embedded offline VIX series, override with `--vix-csv`)
* Calendar decade

Outputs `regimes.csv` per variant. Reveals which strategies post similar Sharpes across regimes (more likely true alpha) vs which collapse in recessions (cyclical exposure).

## Other v3 features

* **Comprehensive metrics table** — Sharpe (net), Sharpe (gross), SR\* (Campbell-Thompson), Max DD, Skew, Kurtosis, OOS R², Mean Turnover, Alpha, t(α). One row per model + ensemble. Saved as `outputs/<variant>/comprehensive.csv`.
* **DM with p-values** — `dm_table.csv` (statistic) **and** `dm_pvalues.csv` (two-sided).
* **Variable importance** — `--mode importance` fits each model on train+val, computes GKX-style zero-set importance, aggregates 920 Kronecker features back to 94 base characteristics. `outputs/<variant>/var_importance.csv`.
* **Streamlit dashboard** — variant selector, comprehensive metrics, DM heatmaps (stat + p-value), portfolio returns, transaction-cost sensitivity, **Forecast Combination** tab, **Regimes** tab, variable importance, paper-vs-improved comparison.

---

## Quick start (command line)

```bash
# ── Paper variant (1957-2016, no TC) ────────────────────────────────
python main.py --mode data-only  --variant paper --wrds-username YOUR_USER
python main.py --mode train      --variant paper --models OLS-3 ENet+H PCR PLS GLM+H
python main.py --mode train      --variant paper --models GBRT+H
python main.py --mode train      --variant paper --models NN1 NN2 NN3 NN4
python main.py --mode evaluate   --variant paper          # also builds ENS-AVG, ENS-MSE
python main.py --mode regimes    --variant paper          # NBER, VIX, decades
python main.py --mode importance --variant paper --models OLS-3 ENet+H PCR PLS GBRT+H

# ── Improved variant (1957-2024, impact-aware TC) ───────────────────
python main.py --mode data-only  --variant improved --wrds-username YOUR_USER
python main.py --mode train      --variant improved --models OLS-3 ENet+H PCR PLS GLM+H
python main.py --mode train      --variant improved --models GBRT+H
python main.py --mode train      --variant improved --models NN1 NN2 NN3 NN4
python main.py --mode evaluate   --variant improved
python main.py --mode regimes    --variant improved
python main.py --mode importance --variant improved --models OLS-3 ENet+H PCR PLS GBRT+H

# ── Dashboard (variant selector in sidebar) ─────────────────────────
streamlit run src/dashboard/app.py
```

The Colab notebook `notebooks/empirical_asset_pricing_ml.ipynb` drives the same flow with cells for runtime restarts and Drive backups.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Tested with Python 3.10, pandas 2.2.2, numpy 1.26.4, pyarrow 15.0.2, torch 2.x with CUDA on Colab T4.

### 2. WRDS credentials

```bash
# One-time setup — saves credentials to ~/.pgpass
python -c "import wrds; wrds.Connection()"

# Or set the environment variable
export WRDS_USERNAME=your_wrds_username
```

### 3. Welch-Goyal macro data

This part is laready done, the file is in `data/`.

Download `PredictorData2023.xlsx` from <https://sites.google.com/view/agoyal145> and place it in `data/` (or pass via `--goyal-csv PATH`).

---

## Project structure

```
empirical_asset_pricing_ml/
├── main.py                       # CLI: data-only / train / evaluate / regimes / importance / dashboard
├── requirements.txt
├── configs/
│   └── experiment.yaml           # Universe, splits, models, costs, portfolio defaults
├── notebooks/
│   └── empirical_asset_pricing_ml.ipynb     # Colab-ready end-to-end
├── src/
│   ├── config.py
│   ├── data/
│   │   ├── wrds_loader.py
│   │   └── characteristics.py
│   ├── models/
│   │   ├── base.py               # ModelBase (ABC)
│   │   ├── ols.py
│   │   ├── elastic_net.py
│   │   ├── pls.py
│   │   ├── gbrt.py
│   │   ├── mlp.py
│   │   └── all_models.py         # Production GKX estimators
│   ├── portfolio/
│   │   ├── construction.py
│   │   ├── costs.py
│   │   └── turnover.py
│   ├── backtest/
│   │   ├── engine.py             # GKX decile backtest + impact-aware TC
│   │   └── walkforward_engine.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── regimes.py
│   │   └── var_importance.py
│   ├── reporting/
│   └── dashboard/
│       └── app.py
├── data/cache/                   # Cached parquet panels and feature matrices
├── outputs/                      # comprehensive.csv, regimes.csv, var_importance.csv, etc.
│   ├── paper/
│   └── improved/
└── logs/
```

---

## Model overview

The full GKX comparison sweep, all sharing the same train / validate / test scaffolding (train 1957–1974, validate 1975–1986, test 1987 onward with annual refits).

| Model    | Type              |
|----------|-------------------|
| OLS-3    | Linear (3 chars)  |
| ENet+H   | Penalised linear with Huber loss |
| PCR      | Principal-component regression   |
| PLS      | Partial least squares            |
| GLM+H    | Group lasso + splines with Huber loss |
| GBRT+H   | Gradient-boosted trees with Huber loss |
| NN1–NN4  | Feed-forward neural nets (depth 1 → 4, geometric pyramid) |
| ENS-AVG  | Equal-weighted ensemble of base models |
| ENS-MSE  | MSE-weighted ensemble of base models   |

Neural nets use BatchNorm + early stopping, adaptive Huber loss (data-driven δ), L1 regularisation on linear weights only, and a 10-seed forecast ensemble per architecture.

---

## Key results

### Paper variant (1957–2016, gross of TC)

| Model    | OOS R² (%) | H-L Sharpe | Max DD (%) | α (%/yr) | t(α) |
|----------|-----------:|-----------:|-----------:|---------:|-----:|
| OLS-3    | 0.090      | 0.89       | 58.5       | 21.5     | 4.87 |
| ENet+H   | 0.070      | 0.76       | 61.6       | 18.7     | 4.19 |
| PCR      | 0.201      | 1.09       | 52.8       | 25.4     | 5.96 |
| PLS      | 0.193      | 1.02       | 59.5       | 27.3     | 5.61 |
| GLM+H    | 0.070      | 0.86       | 60.7       | 21.5     | 4.69 |
| GBRT+H   | 0.347      | 1.35       | 48.3       | 33.2     | 7.41 |
| NN1      | 0.257      | 1.33       | 44.9       | 29.3     | 7.29 |
| NN2      | 0.301      | 1.25       | 51.4       | 31.2     | 6.86 |
| NN3      | 0.457      | 1.38       | 47.0       | 33.2     | 7.54 |
| NN4      | 0.383      | **1.48**   | 44.3       | 35.9     | 8.12 |
| ENS-AVG  | 0.491      | **1.83**   | **35.6**   | **42.9** | **10.02** |
| ENS-MSE  | 0.480      | 1.75       | 37.8       | 42.9     | 9.56 |

### Improved variant (1957–2024, net of impact-aware TC)

| Model    | SR (net) | SR (gross) | OOS R² (%) | Max DD (%) | α (%/yr) | t(α) |
|----------|---------:|-----------:|-----------:|-----------:|---------:|-----:|
| OLS-3    | 0.27     | 0.37       | 0.168      | 84.3       | 7.1      | 1.66 |
| ENet+H   | 0.18     | 0.27       | 0.086      | 91.9       | 5.8      | 1.13 |
| PCR      | 0.32     | 0.42       | 0.145      | 84.5       | 8.8      | 1.96 |
| PLS      | 0.33     | 0.43       | 0.169      | 85.8       | 9.7      | 2.04 |
| GLM+H    | 0.24     | 0.34       | 0.118      | 88.1       | 7.0      | 1.50 |
| GBRT+H   | 0.42     | 0.52       | 0.391      | 81.9       | 12.4     | 2.62 |
| NN1      | 0.38     | 0.48       | 0.227      | 82.1       | 10.6     | 2.36 |
| NN2      | 0.44     | 0.54       | 0.288      | 77.2       | 11.5     | 2.70 |
| NN3      | 0.47     | 0.58       | 0.329      | 75.8       | 12.4     | 2.90 |
| NN4      | 0.47     | 0.57       | 0.416      | 79.5       | 13.6     | 2.91 |
| ENS-AVG  | **0.65** | 0.76       | 0.478      | **68.5**   | **16.9** | **4.01** |
| ENS-MSE  | 0.64     | 0.74       | **0.547**  | 69.6       | 16.9     | 3.92 |

Both variants show the same ranking: linear baselines at the bottom, tree/NN models in the middle, ensembles at the top. The improved variant's lower Sharpes reflect realistic TC drag — most pronounced for linear models (which keep turnover but lose signal-to-noise) and least pronounced for the ensembles.

---

## Features

* Full recursive backtest with no look-ahead bias
* Training / validation / test split (1957–1974 / 1975–1986 / 1987 onward)
* 94+ firm characteristics (Green et al. 2017 subset)
* Kronecker feature expansion (chars × macro = 920 signals) plus 74 industry dummies
* All ML models from the paper
* Huber loss for robust estimation
* Neural networks with BatchNorm, early stopping, 10-seed ensemble
* Diebold-Mariano pairwise model comparison tests
* Long-short decile portfolios (value & equal weighted)
* Campbell-Thompson market-timing Sharpe improvement
* Transaction-cost sensitivity analysis
* Variable importance via zero-set Δ R²
* Forecast-combination ensembles (equal & MSE-weighted)
* Regime-conditional evaluation (NBER, VIX terciles, decades)
* Interactive Streamlit dashboard
* WRDS caching (fast re-runs)

---

## Citation

```bibtex
@article{gu2020empirical,
  title   = {Empirical Asset Pricing via Machine Learning},
  author  = {Gu, Shihao and Kelly, Bryan and Xiu, Dacheng},
  journal = {The Review of Financial Studies},
  volume  = {33},
  number  = {5},
  pages   = {2223--2273},
  year    = {2020},
  publisher = {Oxford University Press}
}
```
