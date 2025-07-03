# VaporIQ Analytics Dashboard

## Overview
A Streamlit dashboard for VaporIQ synthetic dataset featuring:
- Data Visualization
- Classification with multiple models (KNN, Decision Tree, Random Forest, Gradient Boosting)
- Clustering with dynamic k and elbow method
- Association Rule Mining on comma-separated flavor lists
- Regression insights (Ridge, Lasso, Decision Tree Regressor)

## Setup
1. Clone this repo.
2. Ensure the `Data/vaporiq_synthetic_dataset_10k.csv` is present.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run:
   ```
   streamlit run app.py
   ```

## File Structure
```
/Data
  vaporiq_synthetic_dataset_10k.csv
app.py
requirements.txt
README.md
```