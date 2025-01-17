# Insurance Premium Prediction

This project aims to predict insurance premium amounts using a variety of machine learning regression models.  The goal is to minimize the Root Mean Squared Logarithmic Error (RMSLE) on the Playground Series Season 4, Episode 12 dataset.

## Dataset

The project uses the Playground Series Season 4, Episode 12 dataset from Kaggle ([link to dataset](https://www.kaggle.com/competitions/playground-series-s4e12/data)).  This synthetic dataset provides information about insurance policy holders, including demographics, policy details, and past claims.  The target variable is the 'Premium Amount'.

## Approach

The notebook `insurance-amount.ipynb` demonstrates the following workflow:

1. **Data Loading and Exploration:**  Loads the training and test datasets and performs basic exploratory data analysis (EDA) to understand data types, distributions, and missing values.
2. **Missing Value Handling:**
    * Creates new binary features to indicate missingness in 'Annual Income', 'Customer Feedback', and 'Health Score'.
    * Fills missing categorical values with "Unknown".
    * Imputes missing numerical values (except 'Premium Amount') with the median.
3. **Feature Engineering:** Creates new features by combining existing variables:
    * **Numerical-Numerical Interactions:** Ratios and logarithmic transformations of numerical features, such as 'Annual Income_log', 'Health Score_Income log_Ratio', 'Annual Income_Age_Ratio', etc.
    * **Categorical-Categorical Interactions:**  Products of target-encoded categorical variables, like 'Feedback_Marital Status', 'Feedback_Occupation', etc.
    * **Numerical-Categorical Interactions:**  Ratios involving numerical and target-encoded categorical features, like 'Customer Feedback_Credit Score_Ratio'.
4. **Data Splitting:**  Splits the training data into training and validation sets using KFold cross-validation to ensure robust model evaluation. Splits both the target encoded features and categorical features for use in different models.
5. **Model Training and Evaluation:** Trains and evaluates various regression models:
    * CatBoost Regressor
    * CatBoost Classifier (predicting whether premium is above/below a threshold and combining it with a regressor – not fully implemented in current notebook)
    * LightGBM Regressor (with different max_depth settings)
    * LightGBM Classifier (as with CatBoost, not fully implemented)
    * XGBoost Regressor
    * Random Forest Regressor
    * Linear Regression

6. **Target Encoding:** Uses K-fold target encoding to convert categorical variables into numerical while mitigating overfitting.  This is done within the cross-validation loop to avoid data leakage. Also, encodes categorical features using label encoding for models that do not work well with target encoding.

7. **Finding the Best Threshold:**  After the initial logistic regression models, analyzes the predicted probabilities and the true values of `y` and calculates the accuracy, recall, and precision to determine the optimal classification probability threshold.

   




## Results

The performance of the models, measured by RMSLE on the validation set during cross-validation, is shown below:

| Model                                  | RMSLE  |
|---------------------------------------|-------|
| CatBoost Regressor                     | 1.047 |
| LightGBM 1 (max\_depth=5)               | 1.047 |
| LightGBM 2 (max\_depth=9)               | 1.046 |
| LightGBM 3 (max\_depth=15)              | 1.046 |
| LightGBM (Many Features)              | 1.046 |
| XGBoost Regressor                      |  Did not complete due to interruption, requires retraining |
| Random Forest                         |  Did not complete due to interruption, requires retraining |
| Linear Regression                     |  Did not complete due to interruption, requires retraining |


## Further Improvements


* **Missing Data Imputation:** Implement more robust missing value imputation strategies, possibly using K-Nearest Neighbors (KNN) or iterative imputation methods.
* **Feature Scaling:** Employ appropriate scaling or normalization techniques depending on the model used (e.g., StandardScaler for linear models).
* **Hyperparameter Tuning:**  Perform more extensive hyperparameter tuning using GridSearchCV or Bayesian Optimization for all models to potentially further reduce RMSLE. Tuning with early stopping is used in the notebook, but further optimization might be useful.
* **Ensemble and Stacking:** Combine the predictions of multiple models (especially the top performers like LightGBM and CatBoost) using stacking or blending techniques.
* **Classifier Threshold Tuning:** Since a classification model is used to help with some of the regression, tuning the probability threshold to determine when the classification should predict a 1 vs a zero to balance precision and recall should be explored.




## Instructions to Run

1. **Install Libraries:**
   ```bash
   pip install numpy scipy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost category_encoders
