import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from lasso_regression import lasso, scaler

def test_new_data(X_new, y_new=None):
    """
    Test the trained Lasso model on new data.
    
    Parameters:
    -----------
    X_new : array-like, shape (n_samples, n_features)
        New feature data to make predictions on
    y_new : array-like, shape (n_samples,), optional
        True target values for the new data (if available)
    
    Returns:
    --------
    y_pred : array-like, shape (n_samples,)
        Predicted values for the new data
    """
    # Scale the new data using the same scaler
    X_new_scaled = scaler.transform(X_new)
    
    # Make predictions
    y_pred = lasso.predict(X_new_scaled)
    
    # If true values are provided, calculate metrics
    if y_new is not None:
        r2_score = lasso.score(X_new_scaled, y_new)
        mse = np.mean((y_pred - y_new) ** 2)
        print(f"R-squared Score on new data: {r2_score:.4f}")
        print(f"Mean Squared Error on new data: {mse:.4f}")
        
        # Plot actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_new, y_pred, alpha=0.5)
        plt.plot([y_new.min(), y_new.max()], [y_new.min(), y_new.max()], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.show()
    
    return y_pred

def generate_test_data(n_samples=100, n_features=10, noise=0.1, random_state=42):
    """
    Generate new test data with similar characteristics to the training data.
    
    Parameters:
    -----------
    n_samples : int, default=100
        Number of samples to generate
    n_features : int, default=10
        Number of features to generate
    noise : float, default=0.1
        Standard deviation of the noise
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    X_new : array-like, shape (n_samples, n_features)
        Generated feature data
    y_new : array-like, shape (n_samples,)
        Generated target values
    """
    np.random.seed(random_state)
    X_new = np.random.randn(n_samples, n_features)
    y_new = np.dot(X_new, lasso.coef_) + noise * np.random.randn(n_samples)
    return X_new, y_new

if __name__ == "__main__":
    # Example usage
    print("Testing the Lasso model with new generated data...")
    X_new, y_new = generate_test_data()
    y_pred = test_new_data(X_new, y_new)
    
    # Print first few predictions
    print("\nFirst 5 predictions:")
    for i in range(5):
        print(f"Actual: {y_new[i]:.4f}, Predicted: {y_pred[i]:.4f}") 