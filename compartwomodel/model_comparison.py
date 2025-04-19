import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso, Ridge
from typing import Tuple, Dict, Any
import seaborn as sns

def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and prepare the dataset for model comparison.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Features (X) and target (y) arrays
    """
    # Load your dataset here
    # For demonstration, we'll use a synthetic dataset
    np.random.seed(42)
    n_samples, n_features = 100, 20
    X = np.random.randn(n_samples, n_features)
    w = np.random.randn(n_features)
    y = X @ w + np.random.randn(n_samples) * 0.1
    return X, y

def compare_models() -> Dict[str, Dict[str, float]]:
    """
    Compare Lasso and Ridge regression models.
    
    Returns:
        Dict[str, Dict[str, float]]: Dictionary containing performance metrics for each model
    """
    # Load data
    X, y = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models with consistent parameters
    lasso_model = Lasso(alpha=0.1, max_iter=10000, random_state=42)
    ridge_model = Ridge(alpha=0.1, random_state=42)
    
    # Train models
    lasso_model.fit(X_train_scaled, y_train)
    ridge_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    lasso_pred = lasso_model.predict(X_test_scaled)
    ridge_pred = ridge_model.predict(X_test_scaled)
    
    # Calculate metrics
    metrics = {
        'Lasso': {
            'MSE': mean_squared_error(y_test, lasso_pred),
            'R2': r2_score(y_test, lasso_pred),
            'Non-zero coefficients': np.sum(lasso_model.coef_ != 0)
        },
        'Ridge': {
            'MSE': mean_squared_error(y_test, ridge_pred),
            'R2': r2_score(y_test, ridge_pred),
            'Non-zero coefficients': np.sum(ridge_model.coef_ != 0)
        }
    }
    
    # Print comparison
    print("\nModel Comparison Results:")
    print("-" * 50)
    for model, scores in metrics.items():
        print(f"\n{model} Regression:")
        for metric, value in scores.items():
            print(f"{metric}: {value:.4f}")
    
    # Create visualizations
    create_comparison_plots(lasso_model, ridge_model, metrics)
    
    return metrics

def create_comparison_plots(lasso_model: Lasso, ridge_model: Ridge, metrics: Dict[str, Dict[str, float]]) -> None:
    """
    Create comparison plots for the models.
    
    Args:
        lasso_model: Trained Lasso regression model
        ridge_model: Trained Ridge regression model
        metrics: Dictionary containing performance metrics
    """
    # Set style
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Coefficients comparison
    feature_indices = np.arange(len(lasso_model.coef_))
    axes[0, 0].bar(feature_indices - 0.2, lasso_model.coef_, width=0.4, alpha=0.7, label='Lasso')
    axes[0, 0].bar(feature_indices + 0.2, ridge_model.coef_, width=0.4, alpha=0.7, label='Ridge')
    axes[0, 0].set_title('Coefficients Comparison', fontsize=12, pad=10)
    axes[0, 0].set_xlabel('Feature Index', fontsize=10)
    axes[0, 0].set_ylabel('Coefficient Value', fontsize=10)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: MSE comparison
    mse_values = [metrics['Lasso']['MSE'], metrics['Ridge']['MSE']]
    axes[0, 1].bar(['Lasso', 'Ridge'], mse_values, color=['#1f77b4', '#ff7f0e'])
    axes[0, 1].set_title('Mean Squared Error Comparison', fontsize=12, pad=10)
    axes[0, 1].set_ylabel('MSE', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: R2 score comparison
    r2_values = [metrics['Lasso']['R2'], metrics['Ridge']['R2']]
    axes[1, 0].bar(['Lasso', 'Ridge'], r2_values, color=['#1f77b4', '#ff7f0e'])
    axes[1, 0].set_title('R-squared Score Comparison', fontsize=12, pad=10)
    axes[1, 0].set_ylabel('R2 Score', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Non-zero coefficients comparison
    non_zero_values = [metrics['Lasso']['Non-zero coefficients'], metrics['Ridge']['Non-zero coefficients']]
    axes[1, 1].bar(['Lasso', 'Ridge'], non_zero_values, color=['#1f77b4', '#ff7f0e'])
    axes[1, 1].set_title('Number of Non-zero Coefficients', fontsize=12, pad=10)
    axes[1, 1].set_ylabel('Count', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels to the bars
    for ax in [axes[0, 1], axes[1, 0], axes[1, 1]]:
        for i, v in enumerate(ax.containers[0].datavalues):
            ax.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    compare_models() 