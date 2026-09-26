# Baseline Predictive Pipeline -- ETAI

**Beatriz Amaral Fonseca** - 20260619


## Project Overview

This project focuses on developing a machine learning pipeline to predict two-year criminal recidivism (`two_year_recid`) using ProPublica's COMPAS dataset. The primary objective is to evaluate model accuracy while auditing algorithmic bias (*fairness audit*) across demographic groups (`race`), ensuring that race is strictly kept out of training features and used exclusively for auditing.


## Project structure

```
.
├── main.py                  # entry point: run the whole pipeline
├── config.yaml               # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py               # loading
│   ├── data_diagnostics.py   # missingness-mechanism test, domain-rule checks, duplicate check (new week 3)
│   ├── preprocessing.py      # leak-safe cleaning, deployable preprocessing pipeline, and train/test split (week 3 grew this file's job well beyond just the split -- same file, same name as week 2)
│   ├── model.py               # model construction
│   ├── evaluate.py           # accuracy  + fairness check
│   └── results.py            # saves each run's report to disk
├── results/                  # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md              # problem description + full data dictionary
```


## Pipeline progress

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| 3 | EDA + preprocessing -- diagnose the data, then fix it | `src/data_diagnostics.py` (missingness-mechanism test via chi-square + Cramér's V, domain-rule invalid-value detection, two-way duplicate check) and `src/preprocessing.py` (leak-safe category cleanup, mechanism-matched imputation with `_was_missing` indicators for MNAR columns, a deployable `ColumnTransformer`, **and** the train/test split itself, all in the one file rather than split across two) replace the old naive `dropna()`/`pd.get_dummies()` preprocessing; encoder/scaler pair (count encoding + robust scaling) chosen by an empirical grid over 15 repeated splits, checked against the runner-up with a paired comparison so the win isn't just noise; three redundant columns (found via correlation + VIF) dropped; `config.yaml` gains `diagnostics` and `preprocessing` sections -- see "Preprocessing decisions" below. |

---
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

---
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