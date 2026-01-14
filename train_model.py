import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib 

# 1. Load Data
# Ensure the CSV name matches exactly what is in your folder
df = pd.read_csv("Realistic_House_Price_Dataset.csv")

# 2. Clean Columns
df.columns = df.columns.str.lower().str.replace(" ", "")
df = df.drop(columns=['id'])

# 3. Feature Engineering
df['houseage'] = 2024 - df['yearbuilt']
df.drop('yearbuilt', axis=1, inplace=True)

# Add the feature used in App
df['areaperbedroom'] = df['area'] / df['bedrooms']

# 4. Define X and y
X = df.drop('price', axis=1)
y = df['price'] 

# 5. Encoding (One Hot)
X = pd.get_dummies(X, columns=['condition', 'garage', 'location'], drop_first=True)

# 6. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42)

# 7. Scaling
num_cols = ['area', 'bedrooms', 'bathrooms', 'floors', 'houseage', 'areaperbedroom']
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# 8. Train Model
print("Training model...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15]
}

model = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='r2')
model.fit(X_train, y_train)
best_model = model.best_estimator_

# 9. Save Everything (CRITICAL STEP)
joblib.dump(best_model, "house_price_model.pkl", compress=3)
joblib.dump(scaler, "scaler.pkl")          # Save the Scaler
joblib.dump(list(X.columns), "model_columns.pkl") # Save the exact column order

print("✅ Model, Scaler, and Column names saved successfully.")
print(f"Model R2 Score: {model.best_score_:.4f}")

#in terminal run:
#python -m venv venv

#venv\Scripts\activate
#streamlit run app.py