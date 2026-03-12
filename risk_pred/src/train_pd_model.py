from src.utils import BC_UTIL_MEDIAN, REVOL_UTIL_MEDIAN, MAPPED_FILLS, MAPPED_BINS, COLS_IN_PD, WOE_MAPS, apply_woe_transformation, home_owner
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import joblib
import os

PATH = Path.cwd()
DATA_PATH = PATH / "data"
MODELS_PATH = PATH / "models"
PD_MODELS = MODELS_PATH / "pd"
os.makedirs(PD_MODELS, exist_ok=True)


COLS_IN_PD.extend(["target", "probability"])

df = pd.read_csv(DATA_PATH / "destilled_dataset_with_probs.csv", usecols=COLS_IN_PD)
df["revol_util"] = df["revol_util"].fillna(REVOL_UTIL_MEDIAN)
df["bc_util"] = df["bc_util"].fillna(BC_UTIL_MEDIAN)
df["home_ownership"] = df["home_ownership"].apply(home_owner)
df['issue_d'] = pd.to_datetime(df['issue_d'])

train_mask = (df['issue_d'] >= '2013-01-01') & (df['issue_d'] <= '2015-12-31')
test_mask  = (df['issue_d'] >= '2016-01-01') & (df['issue_d'] <= '2016-12-31')
oot_mask   = (df['issue_d'] >= '2017-01-01') & (df['issue_d'] <= '2017-12-31')

df_train = df[train_mask]
df_test = df[test_mask]


df_train = apply_woe_transformation(
    df_train,
    MAPPED_BINS,
    WOE_MAPS,
)

df_test = apply_woe_transformation(
    df_test,
    MAPPED_BINS,
    WOE_MAPS,
)


woe_columns = [col + "_bins" for col in MAPPED_BINS.keys()]
woe_columns.append("home_ownership_bins")

df_train["home_ownership_bins"] = df_train["home_ownership"].map({'MORTGAGE': -0.15704912558200876, 'OWN_OTHER': 0.029727634071044905,'RENT': 0.17035630554360756})
df_test["home_ownership_bins"] = df_test["home_ownership"].map({'MORTGAGE': -0.15704912558200876, 'OWN_OTHER': 0.029727634071044905,'RENT': 0.17035630554360756})

X_train = df_train[woe_columns]
X_test = df_test[woe_columns]

y_train = df_train["target"]
y_test = df_test["target"]

C = 1
best_auc = 0

model = LogisticRegression(
    penalty='l2', 
    C=C, 
    solver='lbfgs', 
    max_iter=1000, 
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)

train_proba = model.predict_proba(X_train)[:, 1]
test_proba = model.predict_proba(X_test)[:, 1]

auc_train = roc_auc_score(y_train, train_proba)
auc_test = roc_auc_score(y_test, test_proba)
gini_test = 2 * auc_test - 1
print(f"{'Train AUC':<12} | {'Test AUC':<12} | {'Gini Test':<12}")
print("_" * 55)
print(f"{auc_train:<12.4f} | {auc_test:<12.4f} | {gini_test:<12.4f}")

if auc_test > best_auc:
    best_auc = auc_test
    best_model = model


print("_" * 55)
print(f"C = 1 | Test AUC = {best_auc:.4f}")

coefficients = pd.DataFrame({
'Feature': ['Intercept'] + [ col.replace("_bins", "") for col in list(X_train.columns)],
'Coeff': [best_model.intercept_[0]] + list(best_model.coef_[0])
})

print(coefficients)
coef = dict(zip(coefficients['Feature'], coefficients['Coeff']))

model_artifacts = {
    "model": best_model,
    "feature_order": woe_columns,
    "mapped_bins": MAPPED_BINS,
    "woe_maps": WOE_MAPS,
    "mapped_fills": MAPPED_FILLS,
    "coefficients": coef,
}

joblib.dump(
    model_artifacts,
    PD_MODELS / "logistic_regression_woe_v1.joblib",
)
