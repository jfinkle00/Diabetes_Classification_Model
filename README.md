# Diabetes_Classification_Model

Author: Jason Finkle

Contact Information: jfinkle00@gmail.com

This project explores predictive modeling for diabetes risk classification using multiple machine learning algorithms, including Random Forest, Logistic Regression, and Support Vector Machines (SVM). The goal is to build interpretable and accurate models to predict diabetes based on patient health indicators such as age, BMI, HbA1c levels, and blood glucose levels.

The repository includes:

Data preprocessing and cleaning steps

Exploratory Data Analysis (EDA) with visualizations

Model training and evaluation (accuracy, AUC, F1-score)

Combined probability scoring across models

Risk tier classification and visual risk distribution plots

This project is designed as a practical example of end-to-end predictive modeling, suitable for educational, portfolio, or healthcare analytics use cases.

Initial Data Exploration

For initial data exploration, I analyzed the distribution of numerical variables to identify any potential outliers. I also ensured the dataset contained no missing or duplicate values that could bias the analysis. To enhance interpretability, I transformed the smoking history variable into an ordinal format:

2 for individuals who currently smoke,

1 for those who previously smoked or are not currently smoking,

0 for those who have never smoked.

Observations with missing or unknown smoking status were excluded from the cleaned dataset to maintain data quality and consistency for model training.

<img width="420" height="336" alt="image" src="https://github.com/user-attachments/assets/c5bf69d3-edaf-4175-9cc0-5d4b61819714" />

<img width="420" height="336" alt="image" src="https://github.com/user-attachments/assets/fdb7bdf3-b3f4-4ce3-a3d5-6dcd7e37b74e" />

Random Forest Model Feature Importance Plot and ROC Curve

<img width="420" height="336" alt="image" src="https://github.com/user-attachments/assets/bf4599c7-5702-4346-8df3-c9ca382e99e2" />

<img width="420" height="336" alt="image" src="https://github.com/user-attachments/assets/93f372bb-5439-485e-8fd4-30c3dc04bb25" />

Overall Model Comparisons After Training and Testing and Diabetes Risk Density Plot

<img width="420" height="336" alt="image" src="https://github.com/user-attachments/assets/d5b815ce-e605-4b03-a765-fd1094f35a50" />

<img width="420" height="336" alt="image" src="https://github.com/user-attachments/assets/d1785aeb-124c-4e72-87ad-942f7965e40d" />

Future Directions

To further improve the robustness and practical application of this project, several future enhancements are planned:

Model Expansion: Explore additional classification algorithms such as Gradient Boosting (XGBoost), LightGBM, and Neural Networks to evaluate performance gains over traditional models.

Hyperparameter Tuning: Perform systematic hyperparameter optimization (e.g., grid search, random search) to improve model performance and reduce overfitting.

Feature Engineering: Incorporate interaction terms, non-linear transformations, or domain-specific features (e.g., BMI categories) to potentially improve predictive accuracy.

Cross-Validation: Implement k-fold cross-validation for more reliable performance estimates and to mitigate potential biases from a single train-test split.

Explainability Tools: Use SHAP values or other model explainability techniques to better interpret individual predictions and overall feature impact.

Deployment Potential: Package the final model into an interactive Shiny app or REST API for user-friendly risk prediction.
