# Red Wine Quality Regression Model

## Overview
This project implements a regression model to predict the quality of red wine based on its physicochemical properties. The quality is treated as a continuous variable, and the model evaluates multiple algorithms to find the best performer. The dataset used is `winequality-red.csv`, containing features like acidity, sugar content, pH, and alcohol level.

## Requirements
To run this project, you need the following Python libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost

Install them using:
```
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

## Setup
1. Ensure the dataset `winequality-red.csv` is in the same directory as the notebook.
2. Open file in Jupyter Notebook or JupyterLab.
3. Run the cells sequentially to execute the code.

## Usage
- **Data Loading and EDA**: The notebook loads the data, performs basic statistics, visualizations, and correlation analysis.
- **Preprocessing**: Features are scaled using StandardScaler.
- **Model Training**: Multiple regression models are trained and evaluated on a train-test split (80/20).
- **Prediction**: Use the trained model to predict wine quality on new samples.

Example prediction code (from the notebook):
```python
# Example new sample
new_sample = [8.3, 0.675, 0.26, 2.1, 0.084, 11.0, 43.0, 0.9976, 3.31, 0.53, 9.2]
new_sample_arr = np.array(new_sample).reshape(1, -1)
new_sample_scaled = scaler.transform(new_sample_arr)
predicted_quality = model.predict(new_sample_scaled)[0]
print("Predicted Wine Quality:", round(predicted_quality, 2))
```

## Results
The following models were evaluated:
- Linear Regression: RMSE = 0.6467, R² = 0.4032
- Ridge Regression: RMSE = 0.6467, R² = 0.4032
- Lasso Regression: RMSE = 0.6471, R² = 0.4025
- Support Vector Regressor: RMSE = 0.6475, R² = 0.4018
- Decision Tree: RMSE = 0.7387, R² = 0.2077
- Random Forest: RMSE = 0.6087, R² = 0.4778
- XGBoost: RMSE = 0.6497, R² = 0.3958
- Neural Network (MLP): RMSE = 0.6475, R² = 0.4018

**Best Model**: Random Forest with R² = 0.4778 and RMSE = 0.6087.

## Model Details
- **Approach**: Data preprocessing includes handling missing values (none in this dataset), feature scaling, and train-test split.
- **Evaluation Metrics**: RMSE (Root Mean Squared Error) and R² score.
- **Best Model Selection**: Based on highest R² score.
- **Visualization**: Bar plots compare RMSE and R² across models.


