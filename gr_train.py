import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =====================
# Load dataset
# =====================
df = pd.read_csv("D:\ML_project2\insurance.csv")
print(df)

# Drop date column
if 'date' in df.columns:
    df.drop(columns=['date'], inplace=True)

# Target and features
X = df.drop('charges', axis=1)
y = df['charges']

# =====================
# Column split
# =====================
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# =====================
# Preprocessing
# =====================
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, numeric_features),
    ('cat', cat_transformer, categorical_features)
])

# =====================
# Gradient Boosting Model
# =====================
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# =====================
# Full Pipeline
# =====================
gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', gb_model)
])

# =====================
# Train-test split
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
gb_pipeline.fit(X_train, y_train)

# =====================
# Evaluation
# =====================
y_pred = gb_pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

# =====================
# Save model
# =====================
with open("insurance_gb_pipeline.pkl", "wb") as f:
    pickle.dump(gb_pipeline, f)
