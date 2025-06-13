# 🏡 House Price Prediction using Linear Regression

This project focuses on predicting house prices using **linear regression** based on housing attributes. It was developed as part of the **Oasis Infobyte Internship Program (OIBSIP)**. The notebook demonstrates end-to-end data analysis, model training, evaluation, and visualization.

---

## 📌 Project Objectives

The primary objectives of this project are:

- **Data Collection**: Use a structured dataset containing numerical and categorical features with `price` as the target variable.
- **Data Exploration and Cleaning**: Analyze dataset structure, handle missing values, and ensure high data quality.
- **Feature Selection**: Identify the most relevant features contributing to house prices.
- **Model Training**: Build a **linear regression model** using Scikit-Learn.
- **Model Evaluation**: Measure model performance using **Mean Squared Error (MSE)** and **R-squared (R²)**.
- **Visualization**: Create plots to show relationships between features and target values, and visualize actual vs predicted prices.

---

## 📁 Dataset Overview

The dataset includes key attributes of residential properties that can influence their market price. Here are the main columns:

- **price**: Sale price of the house *(target variable)*
- **area**: Size of the house in square feet
- **bedrooms**: Number of bedrooms
- **bathrooms**: Number of bathrooms
- **stories**: Number of stories/floors
- **mainroad**: Whether the house is on the main road (`yes`/`no`)
- **guestroom**: Availability of a guest room (`yes`/`no`)
- **basement**: Availability of a basement (`yes`/`no`)
- **hotwaterheating**: Availability of hot water heating (`yes`/`no`)
- **airconditioning**: Availability of air conditioning (`yes`/`no`)
- **parking**: Number of parking spots
- **prefarea**: Whether the house is in a preferred area (`yes`/`no`)
- **furnishingstatus**: Furnishing status (`furnished`, `semi-furnished`, `unfurnished`)

---

## 🔧 Tools & Libraries Used

- **Python** – Programming language
- **Pandas** – Data manipulation and cleaning
- **NumPy** – Numerical computing
- **Matplotlib & Seaborn** – Data visualization
- **Scikit-Learn** – Model training and evaluation
- **Google colab** – Interactive development


## 🧹 Data Exploration & Preprocessing

- **No missing values** detected.
- Applied **one-hot encoding** for categorical variables.
- **Transformed `area`** into `log_area` to account for non-linear scaling, which greatly improved its impact in the model.

---

## ✅ Feature Importance (Coefficients)

| Feature                         | Coefficient | Interpretation |
|----------------------------------|-------------|----------------|
| `log_area`                      | **+0.296**  | Strongest predictor of house price |
| `bathrooms`                     | +0.180      | More bathrooms = higher price |
| `hotwaterheating_yes`          | +0.149      | Enhances comfort and value |
| `basement_yes`                 | +0.137      | Adds usable space |
| `mainroad_yes`                 | +0.115      | Better location access |
| `airconditioning_yes`          | +0.114      | Important comfort factor |
| `furnishingstatus_unfurnished` | −0.111      | Reduces property value |
| `prefarea_yes`                 | +0.106      | Prime location matters |
| `stories`                      | +0.105      | More floors → more space/value |
| `parking`                      | +0.050      | More parking = better price |
| `guestroom_yes`                | +0.047      | Adds to utility |
| `bedrooms`                     | +0.016      | Low impact |
| `furnishingstatus_semi-furnished` | +0.005    | Very little influence |

---

## 🤖 Model Training

- Model: **Linear Regression**
- Tool: **Scikit-Learn**
- Training/Test Split: ~70/30

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```

---

## 📊 Model Evaluation

| Metric          | Value     |
|-----------------|-----------|
| **R² Score**    | ~0.67     |
| **MSE**         | Low       |

- The R² score indicates a decent fit for a linear model.
- MSE shows a reasonable average error between predicted and actual prices.

---

## 📉 Visualizations

- **Correlation Heatmap**: Understand relationships between features.

- **Scatter Plot (Actual vs. Predicted)**:

```python
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
```

---

## 🧠 Key Insights

- **Log transformation of area** significantly improved model accuracy.
- The **log-transformed `area`** is the strongest positive predictor of house price. Key insights include:

- **Bathrooms**, **hot water heating**, and **basement availability** positively impact price by improving comfort and usable space.
- **Main road proximity** and **air conditioning** also significantly increase property value.
- **Unfurnished homes** reduce house prices noticeably.
- **Preferred residential areas** and **more stories** add to the home’s value.
- Features like **bedrooms**, **guest rooms**, **parking**, and **semi-furnished status** show minimal influence on price in this dataset.

In summary, the model indicates that **comfort features, space, and location** are the primary drivers of house pricing.
