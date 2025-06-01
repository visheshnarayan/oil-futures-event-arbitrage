# Predictive Oil Market Strategy Using Soccer Transfers and Anomaly Detection

**Author:** Vishesh Gupta    
**Affiliation:** Apex Fund

---

## Overview

This project examines the relationship between high-profile Saudi soccer transfers and short-term fluctuations in oil markets. By applying machine learning and anomaly detection to Brent and WTI data, we assess which benchmark better anticipates oil squeezes and develop an arbitrage strategy around those signals.

**Key Question:**  
*Which oil benchmark—WTI or Brent—is more predictive of short-term price anomalies during major soccer trades?*


## Objectives

- Determine which oil index (Brent or WTI) signals price squeezes more reliably.
- Use anomaly detection to forecast volatility ahead of player transfers.
- Compare spot and futures market behaviors for short-term arbitrage timing.
- Inform event-driven trading strategy design using machine learning.

---

## Reproducibility

To reproduce the project locally:

```bash
git clone https://github.com/visheshnarayan/oil-futures-event-arbitrage.git
cd oil-futures-event-arbitrage
pip install -r requirements.txt
jupyter notebook vishesh_wti_brent.ipynb
```
## Citation

```
@project{vishesh_oil_signal,
  author = {Vishesh Gupta},
  title = {Event-Driven Arbitrage: Forecasting Oil Market Spikes via Soccer Transfers},
  year = {2024},
  affiliation = {Apex Fund},
  url = {https://github.com/visheshnarayan/oil-futures-event-arbitrage}
}
```
---

## Methodology

### Data Collection and Feature Engineering

- **Sources**: EIA (spot prices), Yahoo Finance (futures), manually curated trade event data.
- **Engineered Features**:  
  - Log-transformed price  
  - Daily returns  
  - Volume changes  
  - Moving averages (3/6/10/365-day)

---

### Machine Learning Pipeline

- **Feature Set**: `["price_log", "volume_change", "difference"]`
- **Models**:
  - `IsolationForest`  
  - `One-Class SVM`  
  - `Facebook Prophet` for segmentation  
  - `PCA` for dimensionality reduction  

Time segmentation was achieved using changepoints from Prophet to reduce signal noise and false positives.

---

## Results

- **WTI consistently signaled anomalies 1–2 days earlier** than Brent around major soccer trades.
- **Futures prices lagged** spot prices, especially in Brent.
- Prophet-based segmentation localized anomaly testing, reducing false positives.
- Anomaly detection aligned closely with Neymar and Mbappe trade dates.

**Anomaly Detection Results**
| Trade | Date       | WTI Signal | Brent Signal | Futures Lag |
|-------|------------|------------|--------------|-------------|
| Neymar | 2017-08-01 | Yes (−1d)  | Weak         | Yes         |
| Mbappe | 2017-08-31 | Yes (−1d)  | Weak         | Yes         |
| Lukaku | 2017-07-14 | Yes (0d)   | Delayed      | Yes         |

---

## Risk and Sensitivity Analysis

- WTI’s U.S. market sensitivity made it a better geopolitical indicator.
- Prophet segmentation minimized impact of global price trends.
- Smaller trades (<$70M) produced weaker market signals, indicating a valuation threshold for model efficacy.

---

## Lessons Learned

- Feature selection (especially volume-based metrics) is more critical than model complexity.
- Prophet time series partitioning improves anomaly detection quality in volatile datasets.
- Market behavior around geopolitical and entertainment events can offer alpha opportunities.

---
