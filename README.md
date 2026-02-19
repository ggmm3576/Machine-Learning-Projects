From Prediction to Profit: A Decision-Optimized Churn Modeling Framework
**1. Business Problem Statement**
This project builds a machine learning system that predicts the probability that a customer will cancel their services (churn).
The objective is not just prediction accuracy, but actionable decision-making. The goal is to identify which customers to contact within a limited outreach budget in order to reduce revenue loss and maximize retained value.
2. Leakage Policy
Strict leakage controls were enforced throughout the project:
Only information available prior to churn was used.
No future data was included.
No test data was used during training.
Out-of-fold (OOF) predictions were used for calibration to prevent target leakage.
The Platt calibration model was trained only on OOF predictions.
This ensures realistic, production-ready performance estimates.
3. Libraries Used
The project loads tools for:
Data cleaning and manipulation: tidyverse, dplyr
Visualization: ggplot2
Modeling: glmnet, caret, ranger, xgboost
Evaluation: ROC, AUC, calibration, Brier score, KS
Feature engineering and preprocessing
Train/test splitting and cross-validation
4. Data Setup and Initial Inspection
Reads the raw CSV file.
Fixes TotalCharges to numeric format.
Inspects:
First rows
Column types
Summary statistics
Dataset dimensions
Missing values
This step ensures structural and quality validation before modeling.
5. Data Type Review and Cleaning
Factor Conversion
Converts categorical text columns into factors.
Checks for incomplete rows.
Diagnosing Missing Rows
Identifies 8 incomplete rows.
All had TotalCharges = NA and tenure = 0.
These customers had just started service and had not yet been billed.
Fixing Missing Values
Replaces TotalCharges = NA with 0 for tenure = 0 customers.
Confirms dataset is now 100% complete.
Post-Fix Validation
Confirms no missing values remain.
Ensures new customers correctly have TotalCharges = 0.
Correlation and Target Exploration
Examines correlations among numeric features.
Compares churn vs non-churn distributions.
Visualizes churn rates across:
Contract type
Internet service
Payment method
Feature Strength Ranking
Variables were ranked using:
Cramér’s V (categorical association with churn)
Cohen’s d (numeric effect size)
AUC (single-variable predictive strength)
6. Feature Preparation and Preprocessing
Organizes features into logical groups.
Cleans category labels.
Orders factor levels properly.
Splits into training and test sets.
Builds a preprocessing recipe that:
Imputes missing values
One-hot encodes categorical variables
Removes zero-variance features
Pre-Model Readiness Checklist
Final validation ensures:
No missing values
Valid factor levels
Reasonable numeric ranges
Proper class balance
Recipe consistency across train and test
Partition Assignment
Applies recipe to training and test sets.
Ensures identical transformed features.
Formats target Churn with:
"No" as first level
"Yes" as second level
7. Baseline and Advanced Models
7a. Base Logistic Regression (GLM)
Trains logistic regression on training data.
Predicts probabilities on test set.
Evaluates:
AUC
Brier score
Confusion matrix
Converts coefficients to odds ratios for interpretation.
7b. Calibrated Random Forest (Platt Scaling)
Trains Random Forest model.
Applies Platt scaling for probability calibration.
Evaluates:
AUC
Precision-Recall
Brier
KS
Confusion matrix
Gain and Lift charts
Displays feature importance.
7c. Gradient Boosted Trees (XGBoost)
Tunes via cross-validation.
Applies Platt calibration.
Evaluates:
AUC
Precision-Recall
Brier
Calibration plots
Confusion matrix
Shows feature importance.
8. Final Production Model: Elastic Net Logistic Regression
This is the production-ready model.
Step-by-Step Process
Uses stratified 5-fold cross-validation.
Selects lambda at alpha = 0.50 based on AUC.
Generates OOF predictions.
Fits Platt calibration on OOF predictions only.
Refits Elastic Net on full training data.
Applies Platt calibration to test predictions.
Final Metrics Computed
AUC (raw and calibrated)
Brier score
Calibration intercept and slope
Precision
Recall
F1
Balanced accuracy
MCC
KS statistic
Artifacts Generated
ROC curve
Precision-Recall curve
Calibration plot
Gain and Lift charts
Confusion matrix heatmap
Metrics summary tables
The full pipeline is saved as an RDS object for reproducible deployment.
9. Final Diagnostics
Evaluates the Platt-calibrated Elastic Net model at threshold 0.50.
Computes:
ROC and AUC
Precision-Recall
Brier
KS
Accuracy
Precision
Recall
F1
Balanced accuracy
MCC
Also generates Gain and Lift charts and prints the confusion matrix.
10. Ablation Study
An ablation study was conducted on the Elastic Net model.
Baseline OOF performance computed using all predictors.
One feature group removed at a time.
Model retrained using identical 5-fold setup.
AUC and Brier changes recorded.
This identifies which feature groups contribute most to predictive performance.
11. Frozen Pipeline and Holdout Scoring
Saves trained transformation function.
Saves tuned Elastic Net model.
Saves Platt calibrator.
If a new holdout dataset is provided:
Applies identical preprocessing.
Generates raw predictions.
Applies calibration.
Outputs final churn probabilities to predictions.csv.
12. Budget-Aware Decision Rule
Transforms predictions into business action.
Inputs:
Churn probabilities
Outreach cost per contact
Expected uplift
Revenue horizon
Discount rate
Outputs per 1,000 customers:
Expected retained revenue
Outreach cost
Net value
Optimal contact threshold
Identifies the outreach level that maximizes profit under budget constraints.
13. OOF Predictions (5-Fold CV)
Creates out-of-fold predictions using 5-fold cross-validation:
Train on 4 folds
Predict on held-out fold
Repeat until all observations receive OOF predictions
These predictions are saved for unbiased evaluation and proper calibration.
