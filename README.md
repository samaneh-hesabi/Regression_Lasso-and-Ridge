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

# 2. Project Structure
```
.
├── lassoscript/          # Lasso regression implementation
├── ridgescript/          # Ridge regression implementation
├── compartwomodel/       # Model comparison scripts
├── model_comparison.png  # Visualization of model comparison
├── requirements.txt      # Python package dependencies
├── environment.yml       # Conda environment configuration
└── README.md            # Project documentation
```

# 3. Dependencies
The project requires the following Python packages:
- numpy==1.24.3: For numerical computations and array operations
- pandas==2.0.3: For data manipulation and analysis
- scikit-learn==1.3.0: For machine learning algorithms and utilities
- matplotlib==3.7.2: For creating static visualizations
- seaborn==0.12.2: For statistical data visualization
- jupyter>=1.0.0: For interactive development and documentation

# 4. Installation
## 4.1 Using pip
```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 4.2 Using conda
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate lassoregression
```

# 5. Usage
## 5.1 Running Individual Models
```bash
# Run Lasso regression
python lassoscript/lasso_regression.py

# Run Ridge regression
python ridgescript/ridge_regression.py
```

## 5.2 Running Model Comparison
```bash
python compartwomodel/model_comparison.py
```

# 6. Results and Analysis
The project generates comprehensive visualizations and analysis comparing Lasso and Ridge regression:

- Coefficient comparison plots
- MSE comparison plots
- R2 score comparison plots
- Non-zero coefficients comparison plots

# 7. Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

# 8. License
This project is licensed under the MIT License - see the LICENSE file for details.

# 9. References
1. Tibshirani, R. (1996). Regression shrinkage and selection via the lasso.
2. Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression.
3. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning.
