<div style="font-size:2.5em; font-weight:bold; text-align:center; margin-top:20px;">Lasso Regression Implementation</div>

## 1. Overview
This directory contains a Python implementation of Lasso Regression, a linear regression technique that performs both variable selection and regularization to enhance prediction accuracy and interpretability of the statistical model. The implementation uses the diabetes dataset from scikit-learn.

## 1.1 Files
- `lasso_regression.py`: Main Python script implementing Lasso Regression
- `README.md`: Documentation file

## 1.2 Features
- Uses the diabetes dataset from scikit-learn
- Implements Lasso Regression using scikit-learn with alpha=0.01
- Performs feature scaling using StandardScaler
- Uses 80-20 train-test split with random_state=42 for reproducibility
- Visualizes feature coefficients with a 12x6 figure size
- Calculates and displays multiple performance metrics:
  - R-squared score
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
- Displays feature importance sorted by coefficient values

## 1.3 Implementation Details
- Dataset:
  - Diabetes dataset from scikit-learn
  - 10 features (age, sex, BMI, blood pressure, etc.)
  - Target variable: disease progression
- Model Parameters:
  - Lasso alpha: 0.01
  - Random state: 42
- Data Split:
  - Training: 80%
  - Testing: 20%
  - Random state: 42
- Visualization:
  - Figure size: 12x6
  - X-axis: Feature Names
  - Y-axis: Coefficient Value
  - Title: Lasso Regression Coefficients for Diabetes Dataset
  - Feature names rotated 45 degrees for better readability

## 1.4 Dependencies
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

## 1.5 Usage
1. Ensure all dependencies are installed
2. Run the main script to train the model:
   ```bash
   python lasso_regression.py
   ```
3. The script will:
   - Load and preprocess the diabetes dataset
   - Train a Lasso model with alpha=0.01
   - Display performance metrics (MSE, RMSE, MAE, R-squared)
   - Show a bar plot of coefficients
   - Print feature importance sorted by coefficient values

## 1.6 Output
- Console output: 
  - Performance metrics (MSE, RMSE, MAE, R-squared)
  - Feature importance table sorted by coefficient values
- Visual output: 
  - Bar plot of Lasso coefficients (12x6 figure)
  - Feature names rotated for better readability

## 1.7 Performance Metrics
The implementation calculates and displays:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared score
- Feature importance based on coefficient magnitudes

## 1.8 Interpretation
- Positive coefficients indicate positive correlation with disease progression
- Negative coefficients indicate negative correlation
- Zero coefficients indicate the feature was not selected by the Lasso model
- The magnitude of coefficients represents the strength of the relationship
- Feature importance is determined by the absolute value of coefficients

## 1.9 Testing with New Data
The `test_lasso_model.py` script provides functionality to:
- Test the trained model with new data
- Generate synthetic test data with similar characteristics
- Calculate and display performance metrics (R-squared, MSE)
- Visualize actual vs predicted values
- Compare predictions with true values (when available)

The script includes two main functions:
1. `test_new_data(X_new, y_new=None)`: Tests the model on new data
2. `generate_test_data(n_samples=100, n_features=10, noise=0.1, random_state=42)`: Generates synthetic test data 