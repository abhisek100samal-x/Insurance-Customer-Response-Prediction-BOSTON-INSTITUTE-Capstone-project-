# Insurance Customer Response Prediction

<p align="center">
  <img src="https://raw.githubusercontent.com/abhisek100samal-x/Insurance-Customer-Response-Prediction/d48b21bfbb0b1dc1becedf75c027231849efbf63/Vehicle%20Insurance%20Analytics%20Dashboard.png"
       alt="Vehicle Insurance Analytics Dashboard" width="900">
</p>

<p align="center">
  <b>End-to-end machine learning project for predicting customer responses to vehicle insurance offers</b>
</p>

---

## 1. Project Overview

Insurance companies often contact large customer groups with policy offers, but only a relatively small proportion respond positively. Broad, untargeted campaigns can therefore consume marketing time and budget without generating proportionate returns.

This project applies exploratory data analysis (EDA) and supervised machine learning to estimate whether a customer is likely to respond positively to a vehicle insurance offer. The intended business use is to help marketing teams prioritize prospects, improve campaign targeting, and make more informed decisions about resource allocation.

**Target variable — `Response`:**

- `1`: Customer responded positively to the insurance offer.
- `0`: Customer did not respond positively.

This is a binary classification problem.

## 2. Business Objectives

- Understand customer demographics, vehicle characteristics, and insurance history.
- Identify patterns associated with positive responses.
- Prepare the data for classification while addressing class imbalance.
- Train and compare multiple classification algorithms.
- Tune model hyperparameters and evaluate generalization.
- Demonstrate how a new customer profile can be scored for response probability.
- Translate analytical findings into practical campaign-targeting insights.

## 3. Dataset

The notebook analyzes **381,109 customer records**. The target distribution is:

| Customer response | Records | Share |
|---|---:|---:|
| No positive response (`0`) | 334,399 | 87.74% |
| Positive response (`1`) | 46,710 | 12.26% |
| **Total** | **381,109** | **100%** |

The target is imbalanced: there are approximately 7.16 non-positive responses for every positive response. A model that predicts only the majority class could achieve high accuracy while failing to identify potential responders. For that reason, accuracy should not be used as the only performance measure.

### Feature reference

| Feature | Description |
|---|---|
| `id` | Unique customer-record identifier |
| `Gender` | Customer gender |
| `Age` | Customer age in years |
| `Driving_License` | Driving-license indicator (`1` = yes, `0` = no) |
| `Region_Code` | Encoded geographical region |
| `Previously_Insured` | Whether the customer already has vehicle insurance |
| `Vehicle_Age` | Vehicle-age category |
| `Vehicle_Damage` | Whether the vehicle was previously damaged |
| `Annual_Premium` | Annual vehicle-insurance premium |
| `Policy_Sales_Channel` | Encoded channel through which the offer is communicated |
| `Vintage` | Number of days the customer has been associated with the company |
| `Response` | Binary target |

> **Data availability:** The notebook reads a local CSV file (`data (capstone).csv`). The raw dataset is not embedded in this README. Add the dataset to your project directory or update the path in the notebook before running it. Confirm that you have permission to redistribute the dataset before publishing it.

## 4. Exploratory Data Analysis

The EDA examines the target distribution, numerical and categorical features, relationships with the target, and feature correlations. Visual analysis uses Matplotlib, Seaborn, and Plotly.

### Selected findings recorded in the notebook

- Customers aged **40–49** have the highest positive response rate in the age-group analysis (approximately 21%); the 30–39 group is also relatively responsive.
- Customers with vehicles older than two years show a higher positive response rate (approximately 29%) than the other vehicle-age groups.
- Customers reporting previous vehicle damage have a substantially higher observed response rate (approximately 24%) than those without previous damage.
- Customers who were not previously insured show a much higher positive response rate than customers who were already insured.
- The notebook reports an overall positive response rate of approximately 12.26%.

These are associations observed in this dataset, not evidence that any feature causes a customer to respond. Response rates can also vary with campaign design, customer mix, and other factors.

## 5. Data Preparation and Feature Engineering

The workflow documented in the notebook includes:

1. Inspecting the data structure, summary statistics, missing values, and duplicate records.
2. Reviewing feature distributions and potential outliers.
3. Separating predictors (`X`) from the target (`y`).
4. Removing selected columns: `id`, `Vintage`, `Driving_License`, and `Annual_Premium`.
5. Encoding categorical variables:
   - Label encoding for binary categories such as `Gender` and `Vehicle_Damage`.
   - One-hot encoding for categorical variables such as `Vehicle_Age` and engineered age groups.
6. Standardizing numerical features using `StandardScaler` where required.
7. Splitting the data into **80% training** and **20% testing** sets with stratification and `random_state=42`.
8. Applying SMOTE to the training data to increase representation of the minority class.

### Leakage prevention

The intended evaluation design is to split the data before fitting data-dependent preprocessing and to apply SMOTE only to the training partition. The test set should retain its original class distribution and remain untouched until evaluation.

For a production-grade implementation, put preprocessing, scaling, and resampling inside an `imblearn` pipeline so that each cross-validation fold learns transformations only from its own training fold. Fit the final pipeline on the full training partition and evaluate it once on the holdout test set.

## 6. Machine Learning Models

Four classifiers are developed and compared:

| Model | Role |
|---|---|
| Logistic Regression | Interpretable baseline for binary classification |
| Decision Tree | Captures nonlinear decision rules |
| Random Forest | Ensemble of decision trees |
| LightGBM | Gradient-boosted tree model |

The notebook also uses `RandomizedSearchCV` with stratified 3-fold cross-validation to explore hyperparameters. ROC-AUC is used as the search scoring metric in the documented tuning sections.

## 7. Evaluation Strategy

The notebook uses several complementary classification metrics:

- **Precision:** Of the customers predicted to respond, how many actually responded?
- **Recall:** Of all customers who responded, how many did the model identify?
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Measures ranking discrimination across classification thresholds.
- **PR-AUC / Average Precision:** Summarizes the precision–recall trade-off and is particularly informative for an imbalanced target.
- **Confusion matrix:** Shows true positives, true negatives, false positives, and false negatives.

The notebook also compares training and test ROC-AUC to inspect generalization.

### Baseline comparison recorded in the notebook

| Model | Train ROC-AUC | Test ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.707 | 0.709 |
| Random Forest | 0.458 | 0.454 |
| Decision Tree | 0.624 | 0.625 |
| LightGBM | 0.205 | 0.202 |

**Interpretation:** Logistic Regression has the strongest reported test ROC-AUC among these baseline comparisons, while the Random Forest and LightGBM scores are below 0.5. AUC values below 0.5 warrant investigation: verify that the positive class is correctly defined, probabilities correspond to `Response = 1`, the feature/target alignment is correct, and the model is configured and evaluated consistently. Do not treat a small train–test gap by itself as proof of a good model; both scores can be consistently poor.

The notebook also tunes the four model families. Re-run the notebook in the intended environment to reproduce the tuned-model results; exact tuned scores are not asserted here.

## 8. Customer-Level Prediction

A prediction helper is included in the notebook to demonstrate scoring an individual customer. It:

1. Converts a customer dictionary into a DataFrame.
2. Creates an `Age_Group` feature.
3. Applies the documented binary category mappings.
4. One-hot encodes vehicle-age and age-group categories.
5. Aligns the columns with the model's training features.
6. Applies the fitted scaler.
7. Returns a predicted class and the estimated probability of a positive response.

The example uses the tuned Random Forest model if it is available in the notebook session.

> **Important implementation note:** The prediction function must use exactly the same feature definitions, encoders, column order, and fitted preprocessing objects used during training. The notebook's prediction example and feature-selection notes should be reconciled before deployment—particularly the handling of `Policy_Sales_Channel` and any columns removed during feature selection. A single serialized preprocessing-and-model pipeline is safer than manually rebuilding transformations.

## 9. Technology Stack

- **Language:** Python
- **Data manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Machine learning:** scikit-learn, LightGBM
- **Imbalanced learning:** imbalanced-learn (SMOTE)
- **Model selection:** `train_test_split`, `StratifiedKFold`, `RandomizedSearchCV`
- **Model persistence utilities:** Joblib

## 10. Repository Structure

A suggested repository layout:

```text
Insurance-Customer-Response-Prediction/
│
├── Insurance Customer Response Prediction.ipynb
├── data/
│   └── data (capstone).csv          # Add locally; not included by default
├── images/
│   └── Vehicle Insurance Analytics Dashboard.png
├── models/                          # Optional: saved model/pipeline artifacts
├── requirements.txt
└── README.md
```

The notebook currently references a local CSV path. Update the loading cell to use a project-relative path, for example:

```python
df = pd.read_csv("data/data (capstone).csv")
```

## 11. Installation and Usage

### Prerequisites

- Python 3.9 or later (use the version supported by your installed package releases)
- Jupyter Notebook or JupyterLab
- The project CSV dataset

### Set up a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib seaborn plotly scikit-learn imbalanced-learn lightgbm catboost joblib jupyter
```

### Launch the notebook

```bash
jupyter notebook
```

Open `Insurance Customer Response Prediction.ipynb`, update the CSV path, and run the cells in order.

> The notebook imports CatBoost in its setup cell. If you retain that import, install it with `python -m pip install catboost`. If CatBoost is not used, remove the unused import instead.

## 12. Business Applications

Subject to validation and operational review, a response-propensity model can support:

- Prioritizing customers for campaign outreach.
- Building customer segments for tailored insurance offers.
- Comparing response patterns across age, vehicle, and insurance-history groups.
- Allocating sales-team effort toward higher-propensity prospects.
- Testing campaign strategies and monitoring response outcomes over time.

A predicted probability is a ranking signal, not a guarantee that a customer will respond. Campaign decisions should account for contact cost, offer economics, customer experience, fairness, and the capacity of the sales team.

## 13. Limitations and Next Steps

- Validate all reported metrics by rerunning the full notebook from a clean kernel.
- Investigate the below-0.5 ROC-AUC results for Random Forest and LightGBM before selecting a final model.
- Use a single leakage-safe preprocessing and modeling pipeline for training, cross-validation, inference, and deployment.
- Select a probability threshold using business costs—especially the relative cost of false positives and false negatives—rather than defaulting automatically to 0.5.
- Evaluate precision–recall performance and calibration on an untouched holdout set.
- Consider temporal or out-of-time validation if campaign dates are available.
- Check subgroup performance and potential fairness concerns before using predictions to make customer-level decisions.
- Monitor performance and data drift after deployment.

## 14. Conclusion

This project demonstrates an end-to-end workflow for insurance customer response prediction, from exploratory analysis and feature preparation to model comparison, hyperparameter tuning, and customer-level scoring. The dataset is notably imbalanced, making minority-class performance and threshold selection central to the business problem.

The analysis provides a foundation for more focused insurance marketing. Before operational use, the final model should be selected on verified holdout results, the preprocessing and inference logic should be consolidated into a reproducible pipeline, and the prediction threshold should be aligned with campaign economics.

---

## Author

**Abhisek Samal**

- GitHub: [abhisek100samal-x](https://github.com/abhisek100samal-x)

---

*This README describes the workflow and findings documented in the project notebook. Re-run the notebook to reproduce results, and verify all metrics and preprocessing steps before using the model in a business setting.*
