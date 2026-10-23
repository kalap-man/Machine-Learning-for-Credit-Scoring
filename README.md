# Machine Learning for Credit Scoring

## Project overview

This project investigates how machine learning can help distinguish creditworthy and higher-risk borrowers using historical consumer lending data. The study compares feature selection methods, approaches to class imbalance, and classification algorithms to identify an effective and interpretable credit-scoring model.

The key finding was that **logistic regression with balanced class weights and 10 selected features achieved a ROC-AUC of approximately 0.722**, performing comparably to or better than more complex models in the final model comparison.

## Dataset

The study used proprietary consumer lending data, covering 2017–2021.

- **Raw data:** approximately 8,500 observations and 5,300 predictors.
- **After cleaning:** 6,080 observations and 344 predictors (303 numerical, 41 categorical).
- **Target:** binary credit outcome (`GOOD = 0`, `BAD = 1`), with approximately 10% of observations in the `BAD` class.

The original dataset is **not included** in this repository because it is proprietary.

## Methodology

1. **Data preparation:** Removed observations with unknown outcomes, standardised missing-value indicators, filtered features and observations with excessive missingness, imputed numerical and categorical values, scaled numerical features, and one-hot encoded categorical variables.
2. **Feature selection:** Compared Information Value, mutual information, F-tests, chi-square tests, recursive feature elimination (RFE), sequential selection, and regularised logistic regression. The selected approach combined F-tests and chi-square filtering with L1-logistic-regression-based RFE, reducing the model to **10 features**.
3. **Class imbalance:** Compared SMOTE, random undersampling, and cost-sensitive learning. **Balanced class weights without resampling** performed best in the reported comparison.
4. **Model development:** Compared logistic and penalised logistic regression, decision-tree-based methods, support vector machines, random forests, and boosting approaches.
5. **Evaluation and interpretation:** Used stratified validation, ROC-AUC as the primary discrimination metric, and accuracy, sensitivity, and specificity as supporting metrics. Examined model behaviour using **SHAP** and partial dependence plots.

## Results

Selected results from the study's final model comparison (ROC-AUC):

| Model | ROC-AUC |
|---|---:|
| **Logistic regression — selected model** | **0.722** |
| Linear SVM | 0.721 |
| L1-penalised logistic regression | 0.720 |
| Random forest | 0.693 |

The selected logistic regression achieved **65.9% accuracy, 65.8% sensitivity, and 66.0% specificity** in the study. Differences in AUC between logistic regression and the penalised logistic models or linear SVM were not statistically significant after the study's multiple-comparison adjustment.

The project illustrates that **feature selection, class-imbalance handling, and interpretability can matter as much as model complexity** in credit risk modelling.

## Project Files

| File | Description |
|---|---|
| `model_testing_final.ipynb` | Main modelling notebook: preprocessing pipeline, feature selection, model configuration, evaluation, and SHAP analysis. |
| `pre_processing.py` | Data cleaning and preparation of the binary target and feature matrix. |

## Reproducibility and limitations

The dataset cannot be shared, so the results cannot be independently reproduced from this public repository alone. The notebook is provided to demonstrate the modelling approach and Python implementation; it is **research code, not a deployable credit decision system**.

The analysis is limited to one proprietary data source with anonymised feature names. The original implementation also performs some preprocessing and feature selection before train/test splitting, which can introduce information leakage. A production-ready extension would fit all transformations within each training fold and require independent, out-of-time validation, calibration, and further checks.
