import polars as pl
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


#load data
df = pl.read_csv('your_matrix.csv')

# Find the target column regardless of its name
possible_names = ["label", "is_phishing"]
target_col = next((name for name in possible_names if name in df.columns), None)

if target_col is None:
    raise ValueError(f"Could not find a label column! Your columns are: {df.columns}")

print(f"Using '{target_col}' as the target column.")

#define x and y
X = df.drop(target_col)
y = df.select(target_col)

#Train 70%, Val 15%, Test 15%
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# init XGBoost
model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    eval_metric="logloss",
    early_stopping_rounds=10 # Prevents overfitting
)

#train
print("Training the trees...")
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

#results
probs = model.predict_proba(X_test)[:, 1]
auc = np.round(roc_auc_score(y_test, probs), decimals=3)

print(f"FINAL PROJECT ACCURACY (AUC): {auc * 100}%")


# example prediction
def get_score(url_data):
    row = pl.DataFrame([url_data]).select(X.columns)
    prob = model.predict_proba(row)[0, 1]
    print(f"\nPhishing Percentage for this URL: {prob * 100:.2f}%")

print("Scoring first URL from test set as example:")
get_score(X_test.to_dicts()[0])
