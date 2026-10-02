# Baseline Predictive Pipeline -- ETAI

**Beatriz Amaral Fonseca** - 20260619


## Project Overview

This project focuses on developing a machine learning pipeline to predict two-year criminal recidivism (`two_year_recid`) using ProPublica's COMPAS dataset. The primary objective is to evaluate model accuracy while auditing algorithmic bias (*fairness audit*) across demographic groups (`race`), ensuring that race is strictly kept out of training features and used exclusively for auditing.

## Pipeline progress

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| 3 | EDA + preprocessing -- diagnose the data, then fix it | `src/data_diagnostics.py` (missingness-mechanism test via chi-square + Cramér's V, domain-rule invalid-value detection, two-way duplicate check) and `src/preprocessing.py` (leak-safe category cleanup, mechanism-matched imputation with `_was_missing` indicators for MNAR columns, a deployable `ColumnTransformer`, **and** the train/test split itself, all in the one file rather than split across two) replace the old naive `dropna()`/`pd.get_dummies()` preprocessing; encoder/scaler pair (target encoding + standard scaling) chosen by an empirical grid over 15 repeated splits, checked against the runner-up with a paired comparison so the win isn't just noise; three redundant columns (found via correlation + VIF) dropped; `config.yaml` gains `diagnostics` and `preprocessing` sections -- see "Preprocessing decisions" below. Threshold-independent metrics (ROC-AUC/PR-AUC) and a calibration check are deliberately **not** added yet -- not yet |
| 4 | Preprocessing inside the pipeline + cross-validation -- evaluating a model honestly | A **locked final test set** (20%, stratified, seed 42) is set aside by `split_dev_test()` (replaces `split_train_test()`) and never scored; models are now judged by **stratified 5-fold cross-validation** of the whole pipeline (preprocessing + model) on the development set, reported per fold with mean ± std and the train-validation gap; the classification report and fairness check now use out-of-fold predictions; target encoding switched to scikit-learn's cross-fitting `TargetEncoder` (a row's own label never leaks into its own encoding), encoder/scaler set by hand in `config.yaml` (target encoding + robust scaling, reasons in the comments); **two fixes** in `clean_dataset()`: genuine `NaN`s in categorical columns were being turned into the string `"nan"` (a fake category), so 229 `c_charge_degree` gaps were never imputed or flagged -- fixed in `config.yaml` alone: `"nan"` added to `diagnostics.placeholder_tokens` (the category cleanup's last step turns listed tokens into `NaN`, after its text conversion); and it no longer drops rows -- de-duplication moved to a separate, training-only `drop_duplicate_rows()` (run before the dev/test split), so the same cleaning can run on new data where every row needs a prediction; `src/data_diagnostics.py` removed -- its one cleaning function (`flag_invalid_values`) moved into `preprocessing.py`, and the EDA-only checks (missingness test, duplicate counts) live in the EDA notebooks, not in every pipeline run; `dummy` (majority-class) model added as the floor to beat, and `random_forest` registered (sensible defaults, untuned); the final model is refit on the whole development set after CV; `config.yaml` gains `test_set` and `cv` sections -- see "Model evaluation" below |


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


### Conclusion

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


### Conclusion

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
| Decision tree | — | 0.610 ± 0.017 | +0.086 |
| Random forest | — | 0.650 ± 0.018 | +0.083 |

The cross-validation accuracy (`0.672`) is much more reliable than the single Week 3 holdout score (`0.699`). A single split score can be overly optimistic simply due to how the rows are divided, while cross-validation tests the pipeline on all development folds and provides a real variance estimate (`±0.013`). The conclusion from Weeks 2 and 3 remains true under cross-validation: Logistic Regression continues to be the best model. It clearly beats the Dummy baseline (`0.549`), gets better validation accuracy than the Decision Tree (`0.610`) and Random Forest (`0.650`), and shows no signs of overfitting (gap of `+0.003` versus over `+0.080` for the tree models).