<div style="font-size:2.5em; font-weight:bold; text-align:center; margin-top:20px;">Lasso and Ridge Regression Comparison</div>

# 1. Project Overview
This project implements and compares two popular regularization techniques in linear regression: Lasso (L1) and Ridge (L2) regression. The implementation includes both theoretical foundations and practical applications, with comprehensive testing and visualization capabilities.

## 1.1 What is Regularization?
Regularization is a technique used to prevent overfitting in machine learning models by adding a penalty term to the loss function. This project focuses on two types of regularization:

### L1 Regularization (Lasso)
- Adds the sum of absolute values of coefficients as penalty
- Tends to produce sparse solutions (many coefficients become exactly zero)
- Useful for feature selection
- Mathematical form: min(||y - Xw||² + α||w||₁)

### L2 Regularization (Ridge)
- Adds the sum of squared coefficients as penalty
- Shrinks all coefficients but rarely makes them exactly zero
- Better for handling multicollinearity
- Mathematical form: min(||y - Xw||² + α||w||₂²)

# 2. Directory Structure
- `lassoscript/`: Contains Lasso regression implementation and tests
  - `lasso_regression.py`: Core implementation of Lasso regression
  - `test_lasso_model.py`: Unit tests for Lasso implementation
  - `coefficients_plot.png`: Visualization of Lasso coefficients
  - `lasso_coefficients.csv`: Saved coefficients data

- `ridgescript/`: Contains Ridge regression implementation and tests
  - `ridge_regression.py`: Core implementation of Ridge regression
  - `test_ridge_model.py`: Unit tests for Ridge implementation
  - `ridge_mse_plot.png`: Visualization of Ridge MSE

- `compartwomodel/`: Contains model comparison scripts and visualizations
  - `model_comparison.py`: Main comparison script
  - `visualization.py`: Helper functions for plotting
  - `utils.py`: Utility functions for data processing

- `model_comparison.md`: Detailed comparison results and analysis
- `requirements.txt`: Project dependencies
- `environment.yml`: Conda environment configuration

# 3. Implementation Details
## 3.1 Lasso Regression Implementation
The Lasso implementation includes:
- Coordinate descent algorithm for optimization
- Cross-validation for hyperparameter tuning
- Feature scaling and standardization
- Coefficient path visualization
- Performance metrics calculation (MSE, R2)

## 3.2 Ridge Regression Implementation
The Ridge implementation includes:
- Closed-form solution using matrix operations
- Cross-validation for hyperparameter tuning
- Feature scaling and standardization
- Learning curve visualization
- Performance metrics calculation (MSE, R2)

## 3.3 Model Comparison Features
The comparison script provides:
- Side-by-side performance comparison
- Coefficient behavior analysis
- Feature importance visualization
- Statistical significance testing
- Cross-validation results

# 4. Dependencies
The project requires the following Python packages:
- numpy==1.24.3: For numerical computations and array operations
- pandas==2.0.3: For data manipulation and analysis
- scikit-learn==1.3.0: For machine learning algorithms and utilities
- matplotlib==3.7.2: For creating static visualizations
- seaborn==0.12.2: For statistical data visualization
- jupyter>=1.0.0: For interactive development and documentation

# 5. Installation and Setup
## 5.1 Using pip
```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 5.2 Using conda
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate lassoregression
```

# 6. Usage
## 6.1 Running Individual Models
```bash
# Run Lasso regression
python lassoscript/lasso_regression.py

# Run Ridge regression
python ridgescript/ridge_regression.py
```

## 6.2 Running Model Comparison
```bash
python compartwomodel/model_comparison.py
```

## 6.3 Running Tests
```bash
# Test Lasso regression
python -m unittest lassoscript/test_lasso_model.py

# Test Ridge regression
python -m unittest ridgescript/test_ridge_model.py
```

# 7. Results and Analysis
## 7.1 Performance Metrics
The comparison evaluates:
- Mean Squared Error (MSE)
- R-squared Score (R2)
- Number of Non-zero Coefficients
- Coefficient Magnitudes

## 7.2 Visualization Outputs
The comparison generates:
- Coefficient comparison plots
- MSE comparison plots
- R2 score comparison plots
- Non-zero coefficients comparison plots

## 7.3 Interpretation Guide
- **Coefficients Plot**: Shows Lasso's sparsity vs Ridge's shrinkage
- **MSE/R2**: Indicates predictive performance
- **Non-zero Coefficients**: Shows feature selection capability

# 8. Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

# 9. License
This project is licensed under the MIT License - see the LICENSE file for details.

# 10. References
1. Tibshirani, R. (1996). Regression shrinkage and selection via the lasso.
2. Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression.
3. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning.
