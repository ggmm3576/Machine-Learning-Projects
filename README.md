**Overview**

This project builds a machine learning system that predicts the probability that a customer will cancel their services, also known as churn. The objective is not simply to maximize predictive accuracy, but to enable actionable decision-making. The model is designed to help a company identify which customers to contact within a limited outreach budget in order to reduce revenue loss and maximize retained value.

Strict leakage controls were enforced throughout the modeling process to ensure realistic, production-ready evaluation. Only information that would be known before a customer cancels was used. No future data or test data was incorporated during training. Out-of-fold predictions were used for calibration to prevent target leakage, and the Platt calibration model was trained only on those out-of-fold scores.

**Data Preparation and Cleaning**
The dataset was first inspected to understand its structure, quality, and completeness. The TotalCharges column was converted to numeric format, and the dataset was examined for column types, summary statistics, dimensions, and missing values.

Eight incomplete rows were identified, all of which corresponded to customers with tenure = 0 and TotalCharges = NA. These customers had just started service and had not yet been billed, which logically explains the missing total charges. Since their total accumulated charges should be zero, those missing values were replaced with 0. After this correction, the dataset was verified to be fully complete with no missing values remaining.

Exploratory analysis was conducted to understand relationships between features and churn. Correlations among numeric variables were examined, churn rates were compared across categorical groups such as contract type and internet service, and feature strength was quantified using Cramér’s V for categorical variables, Cohen’s d for numeric variables, and single-variable AUC scores. This provided an interpretable ranking of predictive strength before modeling.

**Feature Engineering and Preprocessing**
The data was prepared for modeling using a consistent preprocessing pipeline. Categorical variables were converted into factors with properly ordered levels, and the dataset was split into training and test sets. A preprocessing recipe was constructed to impute missing values, one-hot encode categorical variables, and remove zero-variance predictors. This ensured that all models received identical, clean input features.

A pre-model readiness check confirmed that there were no missing values, factor levels were correct, numeric columns were reasonable, and the preprocessing pipeline produced consistent outputs across training and test data. The final training and test matrices were aligned so that both datasets contained identical transformed features, with the target variable Churn encoded consistently.

**Modeling Approach**
Several models were trained and evaluated to compare performance and calibration quality.

A baseline logistic regression model was first fit to establish a reference. It was trained on the training data and evaluated on the test set using AUC, Brier score, and confusion matrix analysis. Coefficients were converted into odds ratios to provide interpretability.

A Random Forest model was then trained and calibrated using Platt scaling to improve probability estimates. Performance was evaluated using AUC, Precision-Recall curves, Brier score, KS statistic, confusion matrices, and Gain and Lift charts. Feature importance was analyzed to identify key drivers of churn.

A Gradient Boosted Tree model using XGBoost was also trained with cross-validation for hyperparameter tuning. Its predicted probabilities were calibrated using Platt scaling, and performance was evaluated using the same comprehensive set of metrics and diagnostic plots.

**Final Production Model: Elastic Net Logistic Regression**
The final production model is a calibrated Elastic Net logistic regression with alpha set to 0.50. Stratified 5-fold cross-validation was used to select the optimal regularization parameter based on AUC. Out-of-fold predictions were generated and used to train a Platt calibration model without introducing leakage. The Elastic Net model was then refit on the full training dataset using the selected lambda, and calibrated predictions were produced for the test set.

Comprehensive evaluation metrics were computed, including raw and calibrated AUC, Brier score, calibration intercept and slope, precision, recall, F1 score, balanced accuracy, Matthews correlation coefficient, and KS statistic. Diagnostic artifacts such as ROC curves, Precision-Recall curves, calibration plots, Gain and Lift charts, confusion matrix heatmaps, and formatted summary tables were generated for reporting.

The full pipeline was saved as an RDS object to allow reproducible deployment and scoring on new data.

**Ablation Study**
An ablation study was conducted to understand the contribution of each feature group. The baseline out-of-fold performance was first computed using all predictors. Then, one feature group at a time was removed, and the model was retrained using the same 5-fold cross-validation setup. Changes in AUC and Brier score were recorded relative to the full model.

This analysis quantifies how much predictive performance depends on each feature group and identifies which categories of variables contribute most to churn prediction.

**Frozen Pipeline and Holdout Scoring**
The final trained pipeline, including the transformation function, tuned Elastic Net model, and Platt calibration model, was saved for reuse. If a new holdout dataset is provided, the identical preprocessing steps are applied, raw predictions are generated, probabilities are calibrated, and final churn probabilities are exported to predictions.csv. This ensures full reproducibility and consistent deployment behavior.

**Business Decision Optimization**
The project goes beyond prediction by incorporating a budget-aware decision framework. Using calibrated churn probabilities, an assumed outreach cost, an estimated uplift from contacting at-risk customers, and a defined revenue horizon, the model estimates expected retained value from targeting the top K customers per 1,000.
For each outreach level, the total outreach cost, expected retained revenue, and net value per 1,000 customers are computed. A capacity sensitivity analysis identifies the contact level that maximizes net profit. This transforms the churn model from a predictive tool into a profit-optimizing decision system.

**Out-of-Fold Predictions**
Out-of-fold predictions were generated using 5-fold cross-validation. In each iteration, the model was trained on four folds and used to predict on the held-out fold, ensuring that every observation received a prediction from a model that had not seen it during training. These OOF probabilities were saved and used for unbiased evaluation and proper calibration.
