# Implied Volatility Surface: SVI Calibration & Machine Learning Extension

A quantitative finance project building a full pipeline from raw AAPL option market data to an interpretable, parametric volatility surface, with a machine learning extension exploring whether a neural network can learn to predict SVI parameters directly from market features.

## What this project does

1. **Retrieves live option chain data** for AAPL via Yahoo Finance and reconstructs a 3D implied volatility surface.
2. **Calibrates the SVI (Stochastic Volatility Inspired) model** to each maturity's smile individually, using numerical optimization (`scipy.optimize`).
3. **Trains a neural network** on synthetically generated smiles to predict SVI parameters directly, and honestly evaluates where this approach works, and where it doesn't, including validation against real market data.
4. **Documents the full analysis** in a reproducible Quarto report (`report.qmd`), which renders to HTML, PDF, and PowerPoint.

## Project structure

```
.
├── vol_surface.py           # Option chain retrieval, surface interpolation, 3D plotting
├── svi_model.py              # SVI formula, calibration, fit-quality diagnostics
├── synthetic_data.py         # Synthetic training data generation for the ML extension
├── train_nn.py                # Neural network training, evaluation, real-market validation
├── report.qmd                 # Quarto report: narrative + function calls only, no analysis logic
├── nn_architecture.png        # Neural network architecture diagram (used in the report)
├── requirements.txt           # Python dependencies
└── README.md
```

The project follows a strict separation of concerns: all computation lives in the `.py` modules as reusable, testable functions; `report.qmd` only imports and calls these functions, plus narrative text. This keeps the analysis logic in one place and the report focused on interpretation.

## Setup

**1. Install Python dependencies:**

```bash
pip install -r requirements.txt
```

**2. Install Quarto** (to render the report), if not already installed:

```bash
brew install quarto        # macOS
```

See [quarto.org](https://quarto.org/docs/get-started/) for other platforms.

## Usage

**Run individual modules directly**, e.g. to see the volatility surface plot or SVI calibration results in isolation:

```bash
python vol_surface.py
python svi_model.py
python synthetic_data.py    # generates svi_training_data.csv
python train_nn.py          # trains the NN, requires svi_training_data.csv to exist first
```

**Render the full report:**

```bash
quarto render report.qmd
```

This produces `report.html` (and, if the PDF/PPTX formats are configured, `report.pdf` / `report.pptx`), re-running every analysis step live: fetching current market data, calibrating SVI, training the neural network, and generating all figures and tables.

## Methodology summary

- **Volatility surface:** built from liquid option quotes (`volume >= 10`), interpolated onto a regular grid for visualization only, SVI calibration itself is performed directly on the filtered market observations.
- **SVI calibration:** per-maturity, minimizing relative squared error in total variance space, restricted to a liquid log-moneyness window. Fit quality (RMSE, relative RMSE) is reported for every maturity. **Note:** static/calendar no-arbitrage constraints are not explicitly enforced; see the report's caveats.
- **Machine learning extension:** since no real multi-day option history was available, training data is synthetically generated with two SVI parameters ($a$, $\rho$) explicitly linked to maturity, and two ($b$, $m$) left independently random, as a controlled test of what the network can and cannot learn. Results are validated against every real AAPL maturity available at render time, not just held-out synthetic data.

## Key findings

- The AAPL volatility surface shows the expected smile shape and a predominantly negative skew ($\rho < 0$ for most maturities), consistent with documented equity crash-risk pricing.
- The neural network recovers real predictive signal only for parameters with a genuine injected relationship to its input features, and is correctly unable to predict parameters left independently random, a useful internal consistency check on the pipeline.
- SVI parameter identifiability is a genuine limitation, empirically demonstrated: recalibrating on noisy data does not reliably recover the true generating parameters for all five parameters equally.
- The trained network does not extrapolate reliably beyond the maturity range seen in training, an important and explicitly documented limitation rather than a hidden one.

## AI assistance disclosure

AI tools (Claude, Anthropic) were used throughout development to debug Python errors, explain concepts interactively, and help structure the analysis and this report. All modelling choices, interpretations, and conclusions were reviewed by the author.

## License

This project is provided for educational purposes.
