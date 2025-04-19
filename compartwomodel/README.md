<div style="font-size:2.5em; font-weight:bold; text-align:center; margin-top:20px;">Model Comparison Analysis</div>

This directory contains scripts and visualizations for comparing Lasso and Ridge regression models in the project.

## 1. Contents

### 1.1 Files
- `model_comparison.py`: Python script that implements and compares Lasso and Ridge regression models
- `model_comparison.png`: Visualization showing the comparison results

## 2. Purpose
The scripts in this directory are designed to:
- Implement and compare Lasso and Ridge regression models using a synthetic dataset
- Compare their performance using multiple metrics:
  - Mean Squared Error (MSE)
  - R-squared (R2) score
  - Number of non-zero coefficients
- Generate comprehensive visualizations of the comparison results
- Help in selecting the most appropriate model for the given dataset

## 3. Features
- Comprehensive model comparison with multiple metrics
- High-quality visualizations including:
  - Coefficient comparison plot
  - MSE comparison
  - R2 score comparison
  - Non-zero coefficients comparison
- Type hints and detailed documentation
- Consistent parameter settings with individual model implementations
- Data preprocessing with standardization
- Random state control for reproducibility

## 4. Usage
To run the model comparison:
```bash
python model_comparison.py
```

The script will:
1. Generate a synthetic dataset
2. Split the data into training and test sets
3. Scale the features using StandardScaler
4. Train both Lasso and Ridge models
5. Calculate and display performance metrics
6. Generate and save comparison visualizations

## 5. Dependencies
The script requires the following Python packages (already included in the project's requirements.txt):
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

## 6. Output
The script generates:
- Console output with detailed performance metrics
- `model_comparison.png` file containing four comparison plots:
  - Coefficients comparison
  - MSE comparison
  - R2 score comparison
  - Non-zero coefficients comparison 