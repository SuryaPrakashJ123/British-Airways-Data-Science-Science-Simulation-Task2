# British Airways: Predictive Modeling of Customer Bookings

## Project Overview

This project develops and interprets a machine learning model to predict the likelihood of a customer completing a flight booking. The goal is to support British Airways with actionable insights for smarter marketing, improved resource planning, and a better customer experience.

---

## Dataset Description

The dataset (`customer_booking (1).csv`) contains detailed records of individual booking requests, including both customer and flight attributes.  
**Key features:**
- `num_passengers`: Number of passengers per booking
- `sales_channel`: Booking method (Internet, Travel Agent, etc.)
- `trip_type`: Round trip or one-way
- `purchase_lead`: Days booked in advance
- `length_of_stay`: Length (days) for round trips
- `flight_hour`: Scheduled departure hour
- `flight_day`: Day of the week
- `route`: Route code (airport pair)
- `booking_origin`: Country of booking
- `wants_extra_baggage`, `wants_preferred_seat`, `wants_in_flight_meals`: Customer preferences
- `flight_duration`: Flight duration in hours
- `booking_complete`: **Target variable (1 if booking completed, 0 otherwise)**

---

## Process Overview

### 1. Data Exploration & Preparation

- Loaded the dataset and explored variables to understand distributions, spot missing values, and check for duplicates.
- Categorical features (`sales_channel`, `trip_type`, `flight_day`, `route`, `booking_origin`) were converted to numeric using one-hot encoding.
- Checked and addressed missing values.
- Removed duplicates as needed.

### 2. Feature Engineering

- Created additional features likely to improve predictions, e.g., total seats, specific customer preferences, or timing bands.
- Retained engineered and relevant original variables for modeling.

### 3. Model Development

- **Random Forest Classifier** selected for strong performance and interpretability.
- Data split into training and test sets (80/20 split).
- Model trained to predict `booking_complete`.

### 4. Model Evaluation

- Model achieved approximately **86% accuracy** on the test set.
- Additional metrics (precision, recall, F1-score) calculated to confirm robust performance.

### 5. Feature Importance Analysis

- Identified the top 10 features influencing bookings, such as:
  - **Purchase lead** (advance booking days)
  - **Flight hour** (departure time)
  - **Length of stay**
  - **Number of passengers**
  - **Flight duration**
  - **Booking origin**
  - **Customer preferences** (meals, seat selection)
- Visualized the top drivers using a bar chart.

---

## Code Summary

<details>
<summary>Click to expand Python workflow</summary>

```python
# 1. Data Loading & Exploration
import pandas as pd
df = pd.read_csv("customer_booking (1).csv", encoding="ISO-8859-1")
print(df.info())
print(df.head())

# 2. Data Cleaning
df = df.drop_duplicates()
df = df.fillna(df.median(numeric_only=True))

# 3. Feature Engineering & Encoding
cat_cols = ['sales_channel', 'trip_type', 'flight_day', 'route', 'booking_origin']
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

X = df_encoded.drop('booking_complete', axis=1)
y = df_encoded['booking_complete']

# 4. Train/Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Random Forest Model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 6. Evaluation
from sklearn.metrics import accuracy_score, classification_report
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 7. Feature Importance
import matplotlib.pyplot as plt
import numpy as np
importances = rf.feature_importances_
indices = np.argsort(importances)[-10:][::-1]
features = X.columns[indices]
plt.figure(figsize=(10,5))
plt.barh(features, importances[indices])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances')
plt.gca().invert_yaxis()
plt.show()


