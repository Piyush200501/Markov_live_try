# Analyzing Stock Market Behavior Using Markov Chains with FII & DII Data

**[→ View the interactive project website](https://Piyush200501.github.io/stock-market-markov-chain/)**

A discrete-time Markov chain model of Nifty 50 regime transitions (Upward /
Downward / Stagnant), conditioned on Foreign and Domestic Institutional
Investor (FII/DII) net flows — built to test whether institutional capital
flows carry predictive information about short-term market regime shifts,
and to challenge the strong-form Efficient Market Hypothesis (EMH).

This repository implements the core methodology from my B.Sc. (Hons)
Mathematics dissertation at Shyam Lal College, University of Delhi
(supervised by Dr. Virender, May 2026).

## Live, Self-Updating Data

The website's interactive sections (state space, conditional matrices,
Monte Carlo simulator, and the "Today's Signal" card) are powered by a
dataset that grows automatically:

- `scripts/update_data.py` fetches the latest Nifty 50 close (via
  [yfinance](https://github.com/ranaroussi/yfinance)) and the latest
  FII/DII net flow (via an unofficial NSE endpoint), appends any new
  trading day(s) to `data/history.json`, recomputes every statistic using
  the exact `src/` pipeline, and rewrites the `REAL_DATA` /
  `LIVE_SIGNAL` objects embedded in `docs/index.html`.
- `.github/workflows/update-live-data.yml` runs this automatically on
  weekday evenings (IST) via GitHub Actions, commits the result, and
  GitHub Pages redeploys within a minute or two — no manual steps.

**"Today's Signal"** is a small live-inference card: it takes the most
recent day's actual state and institutional-flow regime, looks up the
matching row in that regime's conditional transition matrix, and shows
the model's resulting forecast for the next trading day.

### Honest caveats on the automation

- **Nifty 50 price fetch is reliable.** Yahoo Finance's API is
  well-maintained and this is a very standard way to pull index data.
- **FII/DII fetch is the fragile part.** There is no official public API
  for this from NSE or SEBI — `fetch_fii_dii_today()` in
  `scripts/update_data.py` uses an unofficial NSE endpoint that other
  open-source scrapers rely on, but NSE can change this without notice.
  If a scheduled run logs a "FII/DII fetch failed" warning, check
  <https://www.nseindia.com/reports/fii-dii> manually and update the
  parsing logic in that function. This is the one piece of the pipeline
  that will occasionally need a human.
- **Data provenance.** `data/history.json` now stores derived daily
  values (close price, log return, FII/DII net flow, regime label) going
  forward — not full order-book or tick data. This is a policy change
  from the original static build, which deliberately kept raw series out
  of git; committing a slowly-growing daily summary for a personal
  academic/portfolio project is common practice, but you're the one
  publishing it — if you'd rather not, keep `data/history.json` out of
  git and run the script locally only (see `.gitignore`).

### One-time setup to enable auto-push

GitHub Actions can't push commits by default. Enable it once:
**Settings → Actions → General → Workflow permissions → "Read and write
permissions" → Save.** Without this, the workflow will fetch and
recompute correctly but fail on the final `git push` step.

To test it immediately instead of waiting for the schedule: **Actions tab
→ "Update live market data" → Run workflow**.

## Key Findings

| Result | Value |
|---|---|
| Optimal return threshold (sensitivity-tested at 0.20–0.40%) | **0.30%** |
| Diagonal dominance of Upward state (`p₁₁`) — evidence against EMH | **41.29%** |
| Mean First Passage Time (MFPT) back to Upward state | **2.56 days** |
| Downward persistence under Strongly Negative (FII outflow) regime | **41.31%** |
| Bear-market steady-state probability under DII buying (`π_SP`) | **23.06%** |
| Expected bear-market duration: baseline → DII buying regime | 1.703 → **1.276 days** |

**Headline result — "the DII shock absorber":** when domestic institutions
are net buyers, the market's long-run probability of being in a bear
regime collapses from the baseline to 23.06%, and the expected duration
of a downturn shortens substantially. Read as evidence that sustained
SIP-driven domestic liquidity has structurally decoupled Indian equity
markets from pure foreign-flow dependence — with the caveat that this is
one methodology on one market over one time window, not a settled claim
(see Limitations).

**Independent replication:** the live-updating dataset above is itself a
continuously-growing out-of-sample replication window, separate from the
original dissertation's 2018–2024 study period.

## Methodology

1. **State space discretization** — daily Nifty 50 log returns classified
   into Upward / Downward / Stagnant using a symmetric threshold,
   optimized via sensitivity analysis across 0.20–0.40%.
2. **Institutional regime classification** — FII/DII net flows classified
   into Strongly Positive (SP) / Neutral (N) / Strongly Negative (SN)
   using empirical 25th/75th percentiles.
3. **Transition Probability Matrices (TPM)** — unconditional baseline and
   regime-conditioned TPMs estimated via Maximum Likelihood Estimation.
4. **Model validation** — a custom **Top-2 Probabilistic Accuracy** metric
   (a prediction counts as correct if the actual next state falls within
   the model's two most likely predicted states).
5. **Monte Carlo simulation** — 30-day forward path simulation via
   inverse transform sampling.

## Repository Structure

```
stock-market-markov-chain/
├── src/
│   ├── state_discretization.py   # Algorithm 1: state discretization + baseline TPM (MLE)
│   ├── institutional_regimes.py  # Algorithm 2: percentile-based FII/DII regime classification
│   ├── conditional_matrices.py   # Regime-conditioned TPM construction
│   ├── validation.py             # Algorithm 3: Top-2 Probabilistic Accuracy
│   └── monte_carlo_simulation.py # Algorithm 4: MCMC path simulation
├── scripts/
│   ├── update_data.py            # Fetches new data, recomputes, rewrites the site
│   └── requirements.txt          # Extra deps for the update script (pandas, yfinance, requests)
├── data/
│   └── history.json              # Growing daily dataset (see "Live, Self-Updating Data" above)
├── examples/
│   ├── demo.py                   # End-to-end pipeline on synthetic data — runs out of the box
│   └── demo_real_data.py         # Same pipeline on a real Nifty 50 + FII/DII CSV export
├── docs/
│   └── index.html                # Interactive project website (deployed via GitHub Pages)
├── .github/workflows/
│   └── update-live-data.yml      # Daily auto-update job
├── requirements.txt
└── README.md
```

## Getting Started

```bash
git clone https://github.com/Piyush200501/stock-market-markov-chain.git
cd stock-market-markov-chain
pip install -r requirements.txt
python examples/demo.py
```

### Reproducing with real data

```bash
python examples/demo_real_data.py path/to/your_data.csv
```

Or, to run the same live-update pipeline the website uses:

```bash
pip install -r scripts/requirements.txt
python scripts/update_data.py
```

## Tech Stack

Python 3, NumPy (core `src/`); pandas, yfinance, requests (`scripts/`
only). Vanilla HTML/CSS/JS for the website — no build step.

## Limitations

- A first-order Markov assumption is a simplification.
- Findings are specific to the Nifty 50 index over the studied window and
  are not a trading recommendation.
- See "Honest caveats on the automation" above for the live-data pipeline's
  known fragility point.

## Future Work

Continuous-Time Markov Chains (CTMC), Hidden Markov Models (HMM),
higher-order Markov processes, multi-dimensional state space with ML.

## Author

**Piyush Mittal** — B.Sc. (Hons) Mathematics, Shyam Lal College, University
of Delhi. Supervised by Dr. Virender, Department of Mathematics.

## License

MIT — see [LICENSE](LICENSE).
