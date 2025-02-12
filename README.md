# Revenue & Conversions Prediction for SEM Campaigns

## Overview
This project develops a **Linear Regression model** to predict **Revenue and Conversions** for SEM campaigns using a Kaggle dataset. The model is trained with **feature engineering, normalization, encoding**, and includes a simulation example for real-world application.

## Dataset
The dataset originates from Kaggle contains **paid search campaign performance data**, including: impressions, clicks, cost, conversions, revenue, ad group and more.

## Data Preprocessing
To ensure data consistency and optimal model performance, we applied the following preprocessing steps:

### Feature Engineering
- **Cyclical Encoding:** Mapped time-based features (e.g., Month) to a sine-cosine transformation.
- **Feature Engineering:** Created new features like **Cost x Clicks** to capture efficiency trends.

### Normalization
- Used **MinMaxScaler** to scale numerical features between **0 and 1** for model stability.

### Encoding
- Applied **One-Hot Encoding** to categorical variables (e.g., **Device Type, Category**) to convert them into numerical features.

### Consistency Checks
- Ensured the **train, test, and simulation datasets** had identical feature sets and ordering.

## Model Training
A **Multi-Output Linear Regression model** was trained to predict **Revenue and Conversions simultaneously**.

### Key Adjustments for Real-World Predictions:
- **No Negative Predictions:** Conversions and revenue cannot be negative; predictions are clipped at **0**.
- **Logical Dependency Between Revenue & Conversions:**
  - If **Conversions = 0, then Revenue = 0** (ensuring no revenue without conversions).
  - If **Revenue = 0, then Conversions = 0** (ensuring logical consistency).
- **Whole Number Conversions:** Since conversions represent actual transactions, predictions are **rounded** to the nearest whole number.

## Model Performance
### Revenue Prediction
- **Mean Absolute Error (MAE):** 90.94
- **Mean Squared Error (MSE):** 20,411.45
- **R² Score:** 0.9996

### Conversions Prediction
- **Mean Absolute Error (MAE):** 15.00
- **Mean Squared Error (MSE):** 582.94
- **R² Score:** 0.9996

## Error Analysis
Based on validation using the test dataset:
* The majority of conversion predictions fell within a reasonable error range, though some outliers exhibited higher variance.
* 8 out of 30 test samples (27%) had a prediction error greater than 20% for conversions.
* Revenue predictions showed greater variability, suggesting that external factors not captured in the dataset may influence revenue more than conversions.

## Running the Model on Simulation Data
The trained model is applied to a **simulation dataset** to evaluate predictions in a **real-world-like scenario**.

### Required Files
To ensure proper execution, make sure the following files are available:
- `data/simulation_data.csv` – Raw dataset for inference.
- `data/normalized_train_model_final.csv` – Reference dataset for feature alignment.
- `models/joint_revenue_conversions_linear_regression_model.pkl` – Trained model for predictions.
- `scaler/minmax_scaler.pkl` – Pre-fitted scaler for consistent normalization.

## How to Run the Model
1. Clone the repository and upload the required files to your **Colab/Local environment**.
2. Install dependencies (`pandas`, `numpy`, `scikit-learn`, etc.).
3. Load the trained model and MinMaxScaler.
4. Apply feature engineering, normalization, and encoding to simulation data.
5. Run predictions using the trained model.
6. Analyze the predicted **conversions and revenue**.

## Future Improvements
- **Outlier Handling:** Implement **anomaly detection** to filter out high-error predictions.

