<div style="font-size:2.5em; font-weight:bold; text-align:center; margin-top:20px;">Lasso Regression Implementation</div>

## 1. Overview
This directory contains a Python implementation of Lasso Regression, a linear regression technique that performs both variable selection and regularization to enhance prediction accuracy and interpretability of the statistical model.

## 1.1 Files
- `lasso_regression.py`: Main Python script implementing Lasso Regression
- `test_lasso_model.py`: Script for testing the trained model with new data
- `lasso_coefficients.csv`: Output file containing the coefficients of the Lasso model
- `coefficients_plot.png`: Visualization of the Lasso coefficients

## 1.2 Features
- Generates synthetic regression data with 1000 samples and 10 features
- Implements Lasso Regression using scikit-learn with alpha=0.01
- Performs feature scaling using StandardScaler
- Uses 80-20 train-test split with random_state=42 for reproducibility
- Visualizes feature coefficients with a 10x6 figure size
- Calculates and displays R-squared score
- Exports feature importance to CSV
- Generates a bar plot visualization of coefficients

## 1.3 Implementation Details
- Data Generation:
  - 1000 samples
  - 10 features
  - Noise level: 0.1
  - Random state: 42
- Model Parameters:
  - Lasso alpha: 0.01
  - Random state: 42
- Data Split:
  - Training: 80%
  - Testing: 20%
  - Random state: 42
- Visualization:
  - Figure size: 10x6
  - X-axis: Feature Index
  - Y-axis: Coefficient Value
  - Title: Lasso Regression Coefficients
  - Output format: PNG

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
3. To test the model with new data:
   ```bash
   python test_lasso_model.py
   ```
4. The scripts will:
   - Generate and train a Lasso model with alpha=0.01
   - Display the R-squared score
   - Show a bar plot of coefficients
   - Print feature importance
   - Save coefficients to CSV
   - Generate and save a coefficients plot
   - Test the model with new generated data
   - Display actual vs predicted values plot
   - Show performance metrics on new data

## 1.6 Output
- Console output: 
  - R-squared score
  - Feature importance table sorted by coefficient values
- Visual output: 
  - Bar plot of Lasso coefficients (10x6 figure)
  - Saved as `coefficients_plot.png`
- File output: 
  - `lasso_coefficients.csv` containing feature coefficients

## 1.7 Performance Metrics
- R-squared score is calculated on the test set
- Feature importance is determined by the magnitude of coefficients
- The model's performance can be assessed through:
  - R-squared value (closer to 1 indicates better fit)
  - Coefficient sparsity (number of non-zero coefficients)
  - Feature importance ranking

## 1.8 Interpretation
- Positive coefficients indicate positive correlation with the target variable
- Negative coefficients indicate negative correlation
- Zero coefficients indicate the feature was not selected by the Lasso model
- The magnitude of coefficients represents the strength of the relationship 

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