# Baseline Predictive Pipeline -- ETAI

**Beatriz Amaral Fonseca** - 20260619


## Project Overview

This project focuses on developing a machine learning pipeline to predict two-year criminal recidivism (`two_year_recid`) using ProPublica's COMPAS dataset. The primary objective is to evaluate model accuracy while auditing algorithmic bias (*fairness audit*) across demographic groups (`race`), ensuring that race is strictly kept out of training features and used exclusively for auditing.

## Pipeline progress

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| **2** | Introduction & baseline pipeline | Basic end-to-end pipeline with a single train/test split, simple cleaning, Logistic Regression baseline, and an initial false-positive rate fairness check by race. |
| **3** | EDA + preprocessing: diagnose and clean | Replaces naive dropping and one-hot encoding with leak-safe category cleaning, missingness-mechanism indicators (MNAR), and target encoding paired with robust scaling inside a reusable `ColumnTransformer`. |
| **4** | Honest model evaluation & cross-validation | Replaces the optimistic single holdout split with a 20% locked test set and stratified 5-fold cross-validation on the development set, adding the Dummy baseline and registering Random Forest. |
| **5** | Hyperparameter tuning without leakage | Introduces automated tuning with Optuna; instead of static defaults, models are optimized via nested cross-validation to tune parameters honestly without data leakage or optimism bias. |
| **6** | Ensembles in depth | Expands beyond a single tuned tree to ensemble architectures (`oblique_random_forest` and `adaboost`); adds corrected paired t-tests to verify if performance gains exceed cross-validation variance. |


## Week 2

### Results: Logistic Regression vs Decision Tree

| Model | Train Accuracy | Test Accuracy | Gap (train - test) |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | 0.679 | 0.678 | +0.001 |
| **Decision Tree** | 0.829 | 0.629 | +0.200 |


### False Positive Rate (FPR) by Race
*(Share of people who did NOT reoffend, but were predicted to)*

* **African-American (n=303):**
  * Logistic Regression: `FPR = 0.33` | Decision Tree: `FPR = 0.27` | COMPAS: `FPR = 0.44`
* **Caucasian (n=232):**
  * Logistic Regression: `FPR = 0.24` | Decision Tree: `FPR = 0.23` | COMPAS: `FPR = 0.25`

The Decision Tree looked great on training data (0.829), but dropped significantly on test data (0.629). It clearly just memorized the examples instead of truly learning, ending up worse on unseen data than simple Logistic Regression (0.678).


## Week 3

### Preprocessing & Data Cleaning Decisions
Based on the diagnostic findings from `01_eda_introduction.ipynb` and `02_preprocessing.ipynb`, deterministic cleaning rules were integrated into `clean_dataset()` before modeling:

* **Removed Duplicates:** Dropped 72 repeated records across the dataset (`7,286 → 7,214` rows).
* **Domain Rule Enforcements:** Converted implausible entries into `NaN`, including negative juvenile counts, ages outside 18–100, decile scores outside 1–10, and priors counts above 60.
* **Category Canonicalization:** Unified inconsistent casing and whitespace across categorical features (e.g., standardizing demographic variants into `African-American` and `Caucasian`), mapping placeholder tokens like `?` and `-` to `NaN`.


### Results with Cleaned Dataset (Logistic Regression)

| Metric | Train | Test | Gap (train - test) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 0.662 | 0.699 | -0.038 |


### False Positive Rate (FPR) by Race
*(Share of defendants who did NOT reoffend, but were predicted to)*

| Demographic Group | Sample ($N$) | Our Model FPR | COMPAS FPR |
| :--- | :---: | :---: | :---: |
| **African-American** | 289 | 0.27 | 0.42 |
| **Caucasian** | 235 | 0.17 | 0.23 |
| **Hispanic** | 78 | 0.13 | 0.26 |
| **Other** | 44 | 0.07 | 0.14 |
| **Asian** | 5 | 0.20 | 0.20 |

* Implementing `clean_dataset()` eliminated fragmented demographic categories (such as `AFRICAN-AMERICAN`, `African American`, `?`, and `-`), aggregating defendants into consistent cohorts. Test accuracy improved from 0.678 to 0.699 (~70%).
* Despite canonicalizing race, removing corrupted records, and completely excluding race from the model's training inputs, the racial disparity remains: African-American defendants who do not recidivate are still significantly more likely to be falsely predicted as recidivists (`FPR = 0.27`) compared to Caucasian defendants (`FPR = 0.17`).


## Week 4

### Model evaluation

How the pipeline evaluates models, and why (locked test set + stratified 5-fold CV):

A 20% final test set was separated and kept locked to avoid any use during the experimentation phase. On the remaining 80% development set, models are evaluated using stratified 5-fold cross-validation. This setup keeps preprocessing steps fitted strictly inside each training fold, preventing data leakage.

| Model | Holdout accuracy (W3) | CV accuracy (mean ± std) | CV train-val gap |
|---|:---:|:---:|:---:|
| Dummy | — | 0.549 ± 0.000 | +0.000 |
| Logistic regression | 0.699 | 0.672 ± 0.013 | +0.003 |
| Decision tree | 0.629 | 0.610 ± 0.018 | +0.086 |
| Random forest | — | 0.650 ± 0.018 | +0.083 |

The cross-validation accuracy is much more reliable than the single Week 3 holdout score. A single split score can be overly optimistic simply due to how the rows are divided, while cross-validation tests the pipeline on all development folds and provides a real variance estimate. The conclusion from Weeks 2 and 3 remains true under cross-validation: Logistic Regression continues to be the best model. It clearly beats the Dummy baseline, gets better validation accuracy than the Decision Tree and Random Forest, and shows no signs of overfitting.


## Week 5

### Hyperparameter tuning
How the pipeline tunes (Optuna, search space, number of trials, inner folds) and why it doesn't leak.

| Model | Default: CV mean ± std | Tuned: nested CV mean ± std | Tuning score (best trial) | Optimism | Chosen hyperparameters |
|---|---|---|---|---|---|
| Logistic regression | 0.672 ± 0.013 | — | — | — | — |
| Decision tree | 0.610 ± 0.018 | 0.672 ± 0.017 | 0.679 | +0.002 | model__max_depth: 6, model__min_samples_leaf: 101, model__criterion: gini |
| Random forest | 0.650 ± 0.018 | 0.681 ± 0.015 | 0.683 | +0.001 | model__max_depth: 22, model__min_samples_leaf: 42, model__max_features: 0.5 |

The Decision Tree and Random Forest improved significantly with hyperparameter tuning, with both gains clearly exceeding their fold-to-fold standard deviations. Tuning fundamentally changed which model is best: while default Logistic Regression was the initial baseline to beat, the tuned Random Forest overtook it to achieve the highest validation accuracy. For this top performing model, the honest nested CV score of 0.681 should be reported rather than the best trial score to prevent winner's curse optimism bias caused by selecting the maximum across trials.


## Week 6

### Ensembles
| Model | Default: CV mean ± std (gap) | Tuned: nested CV mean ± std (gap) | Tuning score (best trial) | Chosen hyperparameters |
|---|---|---|---|---|
| Logistic regression | 0.672 ± 0.013 | — | — | — |
| Decision tree | 0.610 ± 0.018 | 0.672 ± 0.017 | 0.679 | model__max_depth: 6, model__min_samples_leaf: 101, model__criterion: gini |
| Random forest | 0.650 ± 0.018 | 0.681 ± 0.015 | 0.683 | model__max_depth: 22, model__min_samples_leaf: 42, model__max_features: 0.5 |
| Oblique random forest | 0.647 ± 0.011 | 0.679 ± 0.013 | 0.683 | model__estimator__tree__max_depth: 6, model__estimator__tree__min_samples_leaf: 3, model__estimator__project__n_components: 5 |
| AdaBoost | 0.676 ± 0.012 | 0.679 ± 0.013 | 0.686 | model__estimator__max_depth: 4, model__n_estimators: 115, model__learning_rate: 0.18066055341198425 |

The tuned Random Forest is the strongest candidate, reaching the highest accuracy at 0.681 ± 0.015 with minimal overfitting. Its lead over Oblique Random Forest and AdaBoost is only +0.002, which is smaller than the cross-validation standard deviation and within fold-to-fold variance. Random Forest relies on deeper regularized trees to lower variance, whereas AdaBoost needs shallow trees with a small learning rate to avoid noise. To conclude, Random Forest should be selected, reporting its honest nested CV score rather than the optimistic best trial.

### Challenge - Feature Engineering
| Model | Score without feature engineering | Score with feature engineering | Difference ($\Delta$) |
|---|---|---|---|
| Logistic regression | 0.672 ± 0.013 | 0.676 ± 0.011 | +0.004 | 
| Random forest | 0.681 ± 0.015 | 0.673 ± 0.010 | -0.008 | 
| AdaBoost | 0.679 ± 0.013 | 0.677 ± 0.010 | -0.002 |

Logistic Regression was the only model that improved after feature engineering, while Random Forest and AdaBoost showed slight decreases in performance. Overall, the differences were small relative to the variability across cross-validation folds, suggesting a limited impact of feature engineering. This is consistent with theory, as Logistic Regression can benefit from explicit interaction terms and ratios, whereas tree-based models can capture non-linear relationships and certain interactions directly. However, the observed decreases do not necessarily indicate that the engineered features were redundant or detrimental.