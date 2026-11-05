# 🌸 Task 1 Iris Flower Classification

A beginner-friendly machine learning project that classifies iris flowers into three species (setosa, versicolor, virginica) from four measurements, using **Logistic Regression** and a **Decision Tree**.

Built as part of the Machine Learning & Data Science Internship at Datheon.

## Problem
Given sepal length, sepal width, petal length, and petal width, predict the flower's species.

## Approach
1. Loaded the dataset with Pandas and checked for missing values, duplicates, and class balance
2. Explored the data with a Seaborn pairplot
3. Split the data 80/20 into training and test sets (stratified, `random_state=42`)
4. Scaled features with `StandardScaler` (fit on training data only to avoid data leakage)
5. Trained `LogisticRegression` and `DecisionTreeClassifier`
6. Evaluated both with accuracy, precision, recall, and F1-score
7. Plotted confusion matrices and decision boundaries (petal length vs. petal width)

## Results
| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 0.933 | 0.933 | 0.933 | 0.933 |
| Decision Tree | 0.933 | 0.933 | 0.933 | 0.933 |

Both models got 28 of 30 test flowers right. All mistakes were between versicolor and virginica, whose measurements overlap.

## 📊 Results & Visualisations

### 1️⃣ Exploring the data
<p align="center">
  <img width="75%" alt="Pairplot of the Iris dataset colored by species" src="https://github.com/user-attachments/assets/412ff7b3-9fcd-49f6-8509-06e0df4ef4f6" />
</p>

> Setosa is clearly separated from the other two species, while versicolor and virginica overlap slightly.

---

### 2️⃣ Where do the models go wrong?
<p align="center">
  <img width="90%" alt="Confusion matrices for Logistic Regression and Decision Tree" src="https://github.com/user-attachments/assets/afa8cde9-9fd6-4ed5-b4e6-0eb71fd5a2f4" />
</p>

> Both models classified setosa perfectly and made 2 mistakes each, all between versicolor and virginica.

---

### 3️⃣ How each model separates the classes
<p align="center">
  <img width="95%" alt="Decision boundaries of Logistic Regression and Decision Tree using petal length and petal width" src="https://github.com/user-attachments/assets/c7e74110-3fc5-4eaa-a89f-b2df58b6a684" />
</p>

> Logistic Regression draws smooth, straight borders, while the Decision Tree draws box-shaped regions. Stars are test flowers.
*Pairplot of all four features, coloured by species. Setosa is clearly separated, while versicolor and virginica overlap slightly.*
# Task 2 — Real Estate Housing Price Prediction (OLS vs Ridge vs Lasso)

Predicting median house values on the **California Housing** dataset and comparing **Ordinary Least Squares (OLS)**, **Ridge (L2)** and **Lasso (L1)** regression using **RMSE, MAE and R²**.

Built as part of my Machine Learning & Data Science internship at **Datheon**.

---

## Problem statement

Predict `median_house_value` for California census block groups. The task requires:

- imputing missing values
- engineering interaction features
- handling multicollinearity
- standardising numeric features and encoding categorical ones
- tuning Ridge and Lasso with `GridSearchCV`
- comparing OLS, Ridge and Lasso on RMSE, MAE and R²
- plotting predicted vs actual values and residual distributions

## Dataset

| Property | Value |
|---|---|
| Source | California Housing (1990 census), `housing.csv` |
| Rows × columns | 20,640 × 10 |
| Target | `median_house_value` |
| Categorical feature | `ocean_proximity` |
| Missing values | 207 in `total_bedrooms` |
| Note | Target is capped at $500,001 (965 rows, 4.7%) |

## Approach

1. **Explore** the data: structure, summary statistics, missing values, target distribution.
2. **Split first** (80/20, `random_state=42`) so all preprocessing is learned from training data only (no data leakage).
3. **Feature engineering**
   - Median imputation of `total_bedrooms`
   - Ratios: `rooms_per_household`, `bedrooms_per_room`, `population_per_household` (winsorised at the 0.5/99.5 percentiles)
   - Interactions: `income_x_rooms`, `income_x_age`, `lat_x_lon`, `income_sq` (built from mean-centered values to reduce collinearity)
4. **Multicollinearity**: correlation heatmap and Variance Inflation Factor (VIF), then Ridge/Lasso regularisation.
5. **Preprocessing pipeline**: `StandardScaler` for numeric features, One-Hot encoding for `ocean_proximity`.
6. **Modelling**: OLS, plus Ridge and Lasso tuned over 25 `alpha` values with 5-fold `GridSearchCV`.
7. **Evaluation** on the held-out test set, followed by residual and coefficient analysis.

## Results (test set)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| OLS | $67,418 | $49,044 | 0.653 |
| Ridge (L2) | $67,427 | $49,052 | 0.653 |
| Lasso (L1) | $67,420 | $49,046 | 0.653 |

Best alphas from `GridSearchCV`: Ridge ≈ 1.16, Lasso ≈ 0.68.

### Key findings

- **All three models perform almost identically.** With ~16,500 training rows and 19 features there is little variance for regularisation to remove, so the best alpha is small and the penalty barely changes the predictions.
- **Multicollinearity is real:** `total_bedrooms`, `households`, `total_rooms` and `population` have very high VIFs (up to ~72). Dropping them barely changes accuracy (R² 0.653 → 0.650), so they inflate coefficient variance more than they hurt predictions.
- **Regularisation shows feature importance.** In the Lasso path the tiny `ISLAND` category (5 rows) and the redundant counts are zeroed first, while `median_income`, `bedrooms_per_room` and `ocean_proximity_INLAND` persist.
- **Linear models have limits here.** The $500k cap and a visible non-linearity (cheap areas over-predicted, expensive areas under-predicted) show up in the residual plots.

## 📊 Results & Visualisations

### 1️⃣ Multicollinearity check
<p align="center">
  <img width="75%" alt="Correlation heatmap of numeric features" src="https://github.com/user-attachments/assets/4423a65f-f3e3-410b-8cbe-a08ebbed7217" />
</p>

> `total_rooms`, `total_bedrooms`, `households` and `population` are strongly correlated with each other (VIF up to ~72). This is why regularisation was worth testing.

---

### 2️⃣ Model comparison: OLS vs Ridge vs Lasso
<p align="center">
  <img width="90%" alt="RMSE, MAE and R² comparison across models" src="https://github.com/user-attachments/assets/5dad5e85-ba3b-4a03-a9f4-61a6f08d1c7a" />
</p>

> All three models land on R² ≈ 0.653. The penalty barely matters with ~16,500 rows and 19 features.

---

### 3️⃣ How good are the predictions?
**Predicted vs actual**
<p align="center">
  <img width="95%" alt="Predicted vs actual house values for each model" src="https://github.com/user-attachments/assets/eed1ed9f-a298-4915-89a8-be6a12294c5d" />
</p>

**Residual analysis**
<p align="center">
  <img width="80%" alt="Residual distributions and residual plots" src="https://github.com/user-attachments/assets/cf70000d-46d6-4346-8b8a-c0211e9c7fcc" />
</p>

> The $500k price cap creates a flat ceiling in the points, and the residuals show a clear non-linear pattern: cheap areas are over-predicted, expensive areas under-predicted.

---

### 4️⃣ Regularisation: tuning & feature importance
**Alpha vs RMSE (GridSearchCV)**
<p align="center">
  <img width="95%" alt="Cross-validated RMSE across alpha values for Ridge and Lasso" src="https://github.com/user-attachments/assets/e3798c1d-60ca-4e91-be67-07758354a085" />
</p>

**Lasso coefficient path**
<p align="center">
  <img width="80%" alt="Lasso coefficient path showing features dropping to zero" src="https://github.com/user-attachments/assets/48430d2a-483e-42bc-83ec-0ec270cf3d6f" />
</p>

> Lasso zeroes out the 5-row `ISLAND` category and the redundant count features first, while `median_income`, `bedrooms_per_room` and `ocean_proximity_INLAND` survive the longest.
# Task 3 — Customer Churn Prediction (Random Forest vs XGBoost)
 
Predicting whether a telecom customer will leave the service, comparing **Random Forest** and **XGBoost**, handling class imbalance with **SMOTE** and **class weighting**, and evaluating with **ROC-AUC, precision, recall and F1-score**.
 
Built as part of my Machine Learning & Data Science internship at **Datheon**.
 
---
 
## Problem statement
 
Predict `Churn` (Yes / No) for telecom customers from their subscription details, usage patterns and demographics. The task requires:
 
- cleaning and preprocessing the customer data
- encoding categorical features with `OneHotEncoder`
- handling class imbalance (SMOTE and class weighting)
- training Random Forest and XGBoost classifiers
- 5-fold Stratified Cross-Validation
- tuning hyperparameters with `RandomizedSearchCV`
- evaluating with ROC-AUC, precision, recall and F1-score
- identifying the top 10 features that influence churn
## Dataset
 
| Property | Value |
|---|---|
| Source | Telco Customer Churn, `telco_churn.csv` |
| Rows × columns | 7,043 × 21 |
| Target | `Churn` (Yes / No) |
| Churned / retained | 1,869 / 5,174 (churn rate 26.5%) |
| Numeric / categorical features | 3 / 16 |
| Data issue | 11 blank values in `TotalCharges` |
 
**Cleaning:** `TotalCharges` was converted from text to numeric. The 11 blanks were set to 0 because all of those customers had zero tenure. `customerID` was dropped as it has no predictive value.
 
## Approach
 
1. **Explore** the data: class distribution, churn by contract type, tenure distribution.
2. **Clean**: convert `TotalCharges` to numeric, fix missing values, drop `customerID`.
3. **Split first** (80/20, stratified, `random_state=42`) so all preprocessing is learned from training data only (no data leakage).
4. **Preprocessing pipeline**: `StandardScaler` for numeric features, `OneHotEncoder` for categorical features.
5. **Class imbalance**
   - SMOTE to create synthetic minority-class samples
   - `class_weight="balanced"` for Random Forest
   - `scale_pos_weight` for XGBoost
6. **Modelling**: Random Forest and XGBoost, compared with 5-fold Stratified Cross-Validation.
7. **Tuning**: `RandomizedSearchCV` (12 iterations per model), optimising ROC-AUC.
8. **Evaluation** on the held-out test set with ROC curves, confusion matrices, classification reports and feature importance.
## Results
 
### Class balancing comparison (5-fold CV ROC-AUC)
 
| Model | ROC-AUC |
|---|---|
| XGBoost + Class Weighting | 0.844 |
| XGBoost + SMOTE | 0.843 |
| Random Forest + Class Weighting | 0.823 |
| Random Forest + SMOTE | 0.819 |
 
### Final models (test set, 1,409 customers)
 
| Model | ROC-AUC | Accuracy | Churn Precision | Churn Recall | Churn F1 |
|---|---|---|---|---|---|
| Random Forest | 0.839 | 0.770 | 0.551 | 0.717 | 0.623 |
| XGBoost | 0.844 | 0.774 | 0.560 | 0.701 | 0.622 |
 
Both models beat the target ROC-AUC of 0.80.
 
### Key findings
 
- **Accuracy alone is misleading.** Only 26.5% of customers churned, so ROC-AUC, precision and recall matter more.
- **SMOTE and class weighting perform almost identically.** Their CV ROC-AUC scores differ by only 0.001 to 0.004.
- **XGBoost is slightly better overall** (ROC-AUC 0.844), while Random Forest has slightly higher churn recall (0.717 vs 0.701).
- **Both models catch about 70% of real churners** at the default 0.5 threshold, at the cost of some false alarms (churn precision ≈ 0.55 to 0.56).
- **Main churn indicators:** month-to-month contracts, no online security, no tech support, fiber optic internet, electronic check payments and short tenure.
- **Business insight:** customers on flexible contracts, with short tenure and few add-on services, are most at risk of leaving.
## 📊 Results & Visualisations
 
### 1️⃣ Exploring the data
<p align="center">
  <img width="85%" alt="Class distribution and churn analysis" src="https://github.com/user-attachments/assets/fd13827a-1814-4336-80b5-dc7f8d4565f7" />
</p>
> 1,869 customers churned and 5,174 stayed (26.5% churn), so the classes are imbalanced.
 
---
 
### 2️⃣ Model comparison: ROC curves
<p align="center">
  <img width="70%" alt="ROC curves for Random Forest and XGBoost" src="https://github.com/user-attachments/assets/d7d03fab-8834-4944-be53-c4fe283ad6f0" />
</p>
> XGBoost reaches the highest ROC-AUC (0.844), just ahead of Random Forest (0.839). Both are above the 0.80 target.
 
---
 
### 3️⃣ Where do the models go wrong?
<p align="center">
  <img width="90%" alt="Confusion matrices for Random Forest and XGBoost" src="https://github.com/user-attachments/assets/7423c4ef-08a3-4f18-af9b-aafd1a4af015" />
</p>
> Both models catch about 70% of churners, but also flag some loyal customers as likely to leave (false positives).
 
---
 
### 4️⃣ What drives churn?
<p align="center">
  <img width="90%" alt="Top 10 feature importances for churn prediction" src="https://github.com/user-attachments/assets/0b39f21a-4bd8-4f99-a915-525827d671ae" />
</p>
> Month-to-month contracts are the strongest churn signal, followed by online security, tech support, internet service type, payment method and tenure.
 
---

 
## 💡 Business takeaways
 
Ranking customers by predicted churn probability lets a telecom company target retention early, for example with discounts, better support or longer-term plans for the highest-risk customers.
 
---
## 📁 Repository structure
```
.
├── notebooks_Iris_Classification_Task1.ipynb   # Task 1: Iris classification
├── iris.csv                                    # Task 1 dataset
├── Task2_ regression_housing.ipynb             # Task 2: Housing regression
├── housing.csv                                 # Task 2 dataset (California Housing)
├── Task3_churn_classification.ipynb            # Task 3: Churn classification
├── telco_churn.csv                             # Task 3 dataset (Telco Customer Churn)
├── requirements.txt                            # shared dependencies for all tasks
└── README.md                                   # overview, results and plots
```

## How to run
```bash
pip install -r requirements.txt
jupyter notebook Iris_Classification.ipynb
```

## Tools
Python, Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib

## Author
Hiba Ansari: [LinkedIn](https://www.linkedin.com/in/hiba-ansari24) | [GitHub](https://github.com/hibaansari12)
