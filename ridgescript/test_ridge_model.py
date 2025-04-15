import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import seaborn as sns

def test_ridge_model():
    """
    Test the ridge regression model with a new dataset and evaluate its performance.
    """
    # Generate a new dataset with different characteristics
    X_new, y_new = make_regression(
        n_samples=500,  # Smaller dataset
        n_features=10,
        noise=0.2,      # Higher noise
        random_state=123
    )
    
    # Split the new data
    X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
        X_new, y_new, test_size=0.3, random_state=123
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_new)
    X_test_scaled = scaler.transform(X_test_new)
    
    # Create and train the Ridge model
    ridge = Ridge(alpha=0.01, random_state=123)
    ridge.fit(X_train_scaled, y_train_new)
    
    # Make predictions
    y_pred = ridge.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test_new, y_pred)
    r2 = r2_score(y_test_new, y_pred)
    
    print("\nModel Performance on New Dataset:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared Score: {r2:.4f}")
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_new, y_pred, alpha=0.5)
    plt.plot([y_test_new.min(), y_test_new.max()], 
             [y_test_new.min(), y_test_new.max()], 
             'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.show()
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(ridge.coef_)), ridge.coef_)
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    plt.title('Ridge Regression Coefficients (New Dataset)')
    plt.show()
    
    # Print feature importance
    feature_importance = pd.DataFrame({
        'Feature': [f'Feature_{i+1}' for i in range(len(ridge.coef_))],
        'Coefficient': ridge.coef_
    })
    print("\nFeature Importance:")
    print(feature_importance.sort_values(by='Coefficient', ascending=False))

if __name__ == "__main__":
    test_ridge_model() 