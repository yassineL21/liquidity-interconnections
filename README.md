# Measuring Market Liquidity Interconnections Through VAR Models

## Replication Package for Academic Paper

**Authors**: Youssef Karim El Alaoui (INSEA, Rabat) and Yassine Laarichi (University Ibn Tofail, Kenitra)

**Paper Title**: Measuring Market Liquidity Interconnections Through VAR Models and Impulse Response Analysis: Evidence from Global Financial Markets

---

## Quick Start

This repository contains all code and data necessary to replicate the results in the paper.

### Requirements
- Python 3.10+
- Required packages: `yfinance`, `pandas`, `numpy`, `statsmodels`, `matplotlib`, `seaborn`, `networkx`, `scipy`

```bash
pip install yfinance pandas numpy statsmodels matplotlib seaborn networkx scipy
```

### Data
Daily price and volume data for six major stock indices (1996-2024):
- **US Markets**: Dow Jones (DJIA), S&P 500, NASDAQ
- **European Markets**: FTSE 100, DAX, CAC 40

Data sourced from Yahoo Finance via `yfinance` API.
