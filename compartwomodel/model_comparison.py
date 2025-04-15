import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso, Ridge

# Load and prepare data
def load_data():
    # Load your dataset here
    # For demonstration, we'll use a synthetic dataset
    np.random.seed(42)
    n_samples, n_features = 100, 20
    X = np.random.randn(n_samples, n_features)
    w = np.random.randn(n_features)
    y = X @ w + np.random.randn(n_samples) * 0.1
    return X, y

def compare_models():
    # Load data
    X, y = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    lasso_model = Lasso(alpha=0.1, random_state=42)
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
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Coefficients comparison
    axes[0, 0].bar(range(len(lasso_model.coef_)), lasso_model.coef_, alpha=0.7, label='Lasso')
    axes[0, 0].bar(range(len(ridge_model.coef_)), ridge_model.coef_, alpha=0.7, label='Ridge')
    axes[0, 0].set_title('Coefficients Comparison')
    axes[0, 0].set_xlabel('Feature Index')
    axes[0, 0].set_ylabel('Coefficient Value')
    axes[0, 0].legend()
    
    # Plot 2: MSE comparison
    mse_values = [metrics['Lasso']['MSE'], metrics['Ridge']['MSE']]
    axes[0, 1].bar(['Lasso', 'Ridge'], mse_values, color=['blue', 'orange'])
    axes[0, 1].set_title('Mean Squared Error Comparison')
    axes[0, 1].set_ylabel('MSE')
    
    # Plot 3: R2 score comparison
    r2_values = [metrics['Lasso']['R2'], metrics['Ridge']['R2']]
    axes[1, 0].bar(['Lasso', 'Ridge'], r2_values, color=['blue', 'orange'])
    axes[1, 0].set_title('R-squared Score Comparison')
    axes[1, 0].set_ylabel('R2 Score')
    
    # Plot 4: Non-zero coefficients comparison
    non_zero_values = [metrics['Lasso']['Non-zero coefficients'], metrics['Ridge']['Non-zero coefficients']]
    axes[1, 1].bar(['Lasso', 'Ridge'], non_zero_values, color=['blue', 'orange'])
    axes[1, 1].set_title('Number of Non-zero Coefficients')
    axes[1, 1].set_ylabel('Count')
    
    # Add value labels to the bars
    for ax in [axes[0, 1], axes[1, 0], axes[1, 1]]:
        for i, v in enumerate(ax.containers[0].datavalues):
            ax.text(i, v, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()

if __name__ == "__main__":
    compare_models() 