# 📊 Profitability Intelligence & Predictive Analytics for Superstore

### Identifying the drivers of profitability using statistical modelling, machine learning, and business analytics

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)](https://pandas.pydata.org/)
[![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-Statistical%20Modelling-4051B5)](https://www.statsmodels.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)

---

## 🚀 Project Overview

Retail profitability is influenced by multiple interconnected factors, including sales volume, discounting strategies, product categories, customer behaviour, and geographic performance.

This project analyses the **Superstore transactional dataset** to answer a central business question:

> **What factors most strongly influence profitability, and how can data-driven modelling be used to improve pricing, inventory, and strategic decision-making?**

To answer this, the project combines:

* 🧹 Data cleaning and preprocessing
* 🔍 Exploratory Data Analysis (EDA)
* 📈 Statistical modelling using Multiple Linear Regression
* 🌲 Non-linear predictive modelling using Random Forest Regression
* 📊 Feature importance analysis
* 💼 Business recommendations based on model outputs

The project deliberately combines **interpretability and predictive performance**:

> **Multiple Linear Regression explains the relationships. Random Forest captures the complexity.**

---

## 🎯 Business Objectives

The analysis aims to:

1. Identify the strongest drivers of Superstore profitability.
2. Quantify how sales, discounts, quantity, categories, and regions affect profit.
3. Investigate whether higher sales necessarily translate into higher profitability.
4. Compare an interpretable statistical model with a non-linear machine learning model.
5. Identify the factors that should influence pricing and discount decisions.
6. Provide actionable recommendations for inventory, marketing, and profit optimisation.

---

## 🗂️ Project Structure

```text
📦 superstore-profitability-analysis
│
├── 📄 README.md
├── 📓 Superstore_Profitability_Analysis.ipynb
├── 📊 Superstore.csv
│
└── 📁 visualisations
    ├── category_profit.png
    ├── sales_vs_profit.png
    ├── profit_over_time.png
    ├── correlation_matrix.png
    └── feature_importance.png
```

---

# 🔬 Analytical Workflow

```text
Raw Transactional Data
          ↓
Data Cleaning & Validation
          ↓
Exploratory Data Analysis
          ↓
Feature Engineering & Encoding
          ↓
Multiple Linear Regression
          ↓
Random Forest Regression
          ↓
Model Evaluation & Comparison
          ↓
Feature Importance Analysis
          ↓
Business Recommendations
```

---

# 🧹 1. Data Preparation & Cleaning

The original dataset contained **9,994 transactional records**.

The data preparation pipeline included:

### ✔ Date Parsing

Converted order and shipping dates into datetime objects to enable time-series analysis.

### ✔ Duplicate Removal

Removed duplicate transactions to avoid inflated sales and profit calculations.

### ✔ Numeric Validation

Converted key numerical variables such as:

* Sales
* Profit
* Quantity
* Discount

into appropriate numerical formats.

### ✔ Missing Value Handling

Removed records missing critical business fields required for profitability analysis.

### ✔ Invalid Quantity Filtering

Removed logically invalid transactions with negative quantities.

### ✔ Unique Transaction Validation

Checked the uniqueness of `Row ID` to ensure each transaction was traceable.

### ✔ Extreme Outlier Treatment

Applied the **Interquartile Range (IQR)** method to identify and remove extreme anomalies in:

* Profit
* Sales
* Quantity

After cleaning and outlier treatment:

> **9,994 records → 7,213 analytical records**

This created a more robust dataset for statistical and predictive modelling.

---

# 📊 2. Exploratory Data Analysis

The analysis investigated:

* Profitability by product category
  <img width="269" height="195" alt="image" src="https://github.com/user-attachments/assets/50ca5ed1-87b9-4768-8151-0bce166df632" />

* Profit by state
  <img width="389" height="239" alt="image" src="https://github.com/user-attachments/assets/b658a225-98d9-4b99-889d-3432d046956d" />

* Sales by region
  <img width="262" height="190" alt="image" src="https://github.com/user-attachments/assets/8cc7272d-3b81-45e8-82a7-19494dd109f7" />

* Shipping method usage
  <img width="277" height="190" alt="image" src="https://github.com/user-attachments/assets/c3efc0b1-ff19-4374-98ca-824dc7983e8d" />

* Sales versus profit relationships
  <img width="255" height="188" alt="image" src="https://github.com/user-attachments/assets/1902c472-c458-4df9-8935-d7874c30706a" />

* Profitability trends over time
  <img width="346" height="185" alt="image" src="https://github.com/user-attachments/assets/910695f5-8d84-4c03-920f-f57079893214" />

* Correlations between key variables
<img width="303" height="241" alt="image" src="https://github.com/user-attachments/assets/cd0070fd-b420-4d31-b85f-a98ef8c550e2" />

  <img width="380" height="293" alt="image" src="https://github.com/user-attachments/assets/23f483e1-99a6-4b58-b67f-1d55fbbf97ed" />

* Covariance between numerical features
  <img width="299" height="251" alt="image" src="https://github.com/user-attachments/assets/50585ee9-7ca3-4854-aaf4-b815e9232757" />


## 🔎 Key EDA Insights

### 💰 Sales do not automatically mean profitability

A positive relationship exists between Sales and Profit, but the relationship is far from perfect.

Several high-sales transactions generated negative or low profit.

This highlights an important business insight:

> **Revenue growth without margin control can destroy profitability.**

---

### 🏷️ Discounts are strongly associated with lower profit

Discount demonstrated a substantial negative relationship with Profit.

This suggests that aggressive discounting may increase sales volume while significantly reducing margins.

---

### 📦 Quantity is a relatively weak profitability driver

Quantity showed only a weak relationship with Profit, suggesting that simply selling more units does not guarantee improved profitability.

---

### 🌍 Regional and category-level differences matter

Profitability varies significantly across product categories and geographic regions, suggesting that a single pricing or inventory strategy may not be optimal for the entire business.

---

# 📈 3. Multiple Linear Regression

The first model used **Ordinary Least Squares (OLS) Multiple Linear Regression**.

The objective was not only to make predictions but also to understand the directional relationship between business variables and Profit.

### Model Concept

```text
Profit =
β₀
+ β₁(Sales)
+ β₂(Discount)
+ β₃(Quantity)
+ β₄(Category)
+ β₅(Sub-Category)
+ β₆(Region)
+ ε
```

### Why Multiple Linear Regression?

MLR provides:

* Coefficients
* Direction of influence
* Statistical significance
* P-values
* Confidence intervals
* Interpretability

This makes it particularly useful for explaining business relationships to stakeholders.

## 📊 MLR Results

| Metric      | Result |
| ----------- | -----: |
| Training R² |  0.625 |
| Test R²     |  0.563 |
| Test RMSE   |  10.94 |

### Key findings

* Sales had a strong positive relationship with Profit.
* Discount had a strong negative relationship with Profit.
* Quantity had a comparatively smaller impact.
* The model provided useful interpretability but struggled to capture complex non-linear interactions.

---

# 🌲 4. Random Forest Regression

Retail profitability is unlikely to follow purely linear relationships.

For example:

* The effect of a discount may depend on the product category.
* The effect of sales may differ by region.
* The relationship between quantity and profit may change depending on the product.

To capture these complex relationships, a **Random Forest Regressor** was implemented.

### Why Random Forest?

Random Forest can:

* Capture non-linear relationships
* Model feature interactions
* Handle complex decision boundaries
* Reduce dependence on linear assumptions
* Provide feature importance estimates

## 📊 Random Forest Results

| Metric | Multiple Linear Regression | Random Forest |
| ------ | -------------------------: | ------------: |
| R²     |                      0.563 |     **0.819** |
| RMSE   |                      10.94 |      **7.06** |
| MAE    |                          — |      **3.19** |

### Performance Improvement

The Random Forest model:

* Improved R² by approximately **0.256**
* Reduced RMSE by approximately **35.4%**
* Explained approximately **82% of the variance in Profit**

This demonstrates that profitability is influenced by complex, non-linear relationships that cannot be fully captured by a linear model.

---

# 🧠 5. Feature Importance

The Random Forest model identified the following key drivers:

| Feature                  | Importance |
| ------------------------ | ---------: |
| Sales                    |     47.98% |
| Discount                 |     28.57% |
| Storage Sub-Category     |      5.18% |
| Office Supplies Category |      3.85% |
| Quantity                 |      3.34% |

## 💡 Main Insight

The two dominant factors were:

> **Sales + Discount = approximately 76.5% of the model's feature importance**

This provides a clear strategic message:

> **The business should focus not only on increasing sales, but on increasing profitable sales.**

---

# 💼 6. Business Recommendations

## 1. Implement Discount Governance

Discounting was identified as one of the strongest drivers of profitability.

Recommended actions:

* Establish discount thresholds.
* Monitor high-discount transactions.
* Restrict deep discounts to carefully selected product categories.
* Evaluate discounts based on margin rather than revenue alone.

---

## 2. Prioritise High-Performing Categories

The analysis identified strong-performing categories and sub-categories that can receive greater focus in:

* Inventory planning
* Marketing
* Product placement
* Cross-selling campaigns

---

## 3. Re-evaluate Consistently Underperforming Products

Products and categories generating persistent losses should be analysed for:

* Pricing problems
* Excessive discounting
* High operational costs
* Supplier costs
* Low demand
* High return rates

---

## 4. Focus on Profitable Sales Growth

Since Sales was the strongest predictive feature, increasing sales remains important.

However, the analysis shows that sales growth should be achieved through:

* Product bundles
* Loyalty programmes
* Cross-selling
* Targeted promotions
* Higher-margin product strategies

rather than simply increasing discounts.

---

## 5. Investigate Regional Profitability Differences

Regional variations should be investigated through:

* Logistics costs
* Shipping delays
* Return rates
* Regional pricing
* Distribution costs

A region generating high sales but lower profit may represent an operational optimisation opportunity.

---

## 6. Build a Profitability Intelligence Dashboard

The modelling pipeline could be extended into a business dashboard containing:

* Profit by category
* Profit by region
* Discount impact analysis
* Loss-making product identification
* Profit forecasts
* Sales and margin monitoring
* What-if pricing scenarios

---

# 🧪 7. Model Comparison

| Capability               | Multiple Linear Regression | Random Forest |
| ------------------------ | -------------------------- | ------------- |
| Interpretability         | ⭐⭐⭐⭐⭐                      | ⭐⭐⭐           |
| Non-linear relationships | ⭐                          | ⭐⭐⭐⭐⭐         |
| Feature interactions     | Limited                    | Strong        |
| Predictive performance   | Moderate                   | High          |
| Business explanation     | Excellent                  | Moderate      |
| Profit prediction        | Moderate                   | Strong        |

### Final Modelling Strategy

The strongest approach is not to choose one model exclusively.

Instead:

> **Use Multiple Linear Regression to understand the business.**
>
> **Use Random Forest to predict the business.**

This combination provides both:

* **Explainability for decision-makers**
* **Predictive power for future planning**

---

# 🛠️ Technologies Used

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Visualisation

* Matplotlib
* Seaborn
* Plotly

### Statistical Modelling

* Statsmodels
* Ordinary Least Squares (OLS)
* Multiple Linear Regression

### Machine Learning

* Scikit-learn
* Random Forest Regression

### Environment

* Jupyter Notebook
* Deepnote

---

# 📚 Dataset

The project uses the **Superstore transactional dataset**, containing retail order-level information including:

* Sales
* Profit
* Discount
* Quantity
* Product Category
* Product Sub-Category
* Customer Segment
* Region
* State
* Shipping Mode
* Order Date
* Shipping Date

---

# 📌 Key Takeaways

### 🔹 Revenue is not the same as profitability

High sales do not guarantee high profits.

### 🔹 Discounting is a major profitability lever

Aggressive discounts can significantly reduce margins.

### 🔹 Profitability is non-linear

Random Forest substantially outperformed the linear model, indicating complex interactions between business variables.

### 🔹 Explainability and prediction serve different purposes

Statistical modelling helps explain **why** profitability changes.

Machine learning helps predict **what is likely to happen next**.

---

# 📈 Future Improvements

Future versions of this project could include:

* Hyperparameter optimisation using GridSearchCV or RandomizedSearchCV
* Cross-validation
* SHAP explainability
* XGBoost and Gradient Boosting comparison
* Time-series profit forecasting
* Customer-level profitability modelling
* Price elasticity analysis
* Profit optimisation simulations
* Interactive Power BI or Tableau dashboard
* Deployment as a Streamlit application
* Automated model retraining pipeline

---

# 👤 Author

### Atharv Kothari

MSc Business Analytics & Consultancy
Heriot-Watt University, Edinburgh

B.Tech Information Technology
Vellore Institute of Technology

Interested in:

* Data Science
* Machine Learning
* Business Analytics
* AI Applications
* Predictive Modelling
* Data-Driven Decision Making

---

## ⭐ If you found this project interesting, feel free to explore the notebook and analysis.

> **From descriptive analytics to predictive intelligence — this project demonstrates how business data can be transformed into actionable profitability strategy.**
