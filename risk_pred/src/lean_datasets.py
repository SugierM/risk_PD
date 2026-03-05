from pathlib import Path
import pandas as pd
from ydata_profiling import ProfileReport
import json
import os

"""
1. Cuts unnecessary columns and creates datasets for training, test, and OOT.
2. Create two reports using ydata_profiling
"""

PATH = Path.cwd()
DATA_PATH = PATH / "data"
REPORTS_PATH = PATH / "reports"
os.makedirs(REPORTS_PATH, exist_ok=True)

COLUMNS = ["loan_amnt", "term", "installment", "purpose", "annual_inc", "loan_status",
        "emp_length", "home_ownership", "dti", "delinq_2yrs", "mths_since_last_delinq", "pub_rec",
        "inq_last_6mths", "open_acc", "total_acc", "revol_bal", "revol_util", "acc_now_delinq",
        "mort_acc", "earliest_cr_line", "avg_cur_bal", "num_tl_op_past_12m", "pct_tl_nvr_dlq",
        "chargeoff_within_12_mths", "mo_sin_old_rev_tl_op", "tot_cur_bal",
        "bc_util", "num_actv_bc_tl", "num_tl_90g_dpd_24m", "tot_coll_amt",
        "total_rev_hi_lim", "initial_list_status", "inq_fi", "inq_last_12m",
        "issue_d", "application_type"
]

COLS_IN_PD = ['annual_inc', 'target', 'num_tl_op_past_12m', 'inq_fi_6m',
 'home_ownership', 'tot_cur_bal', 'total_rev_hi_lim', 'bc_util', 'dti',
 'mo_sin_old_rev_tl_op', 'revol_util', 'loan_amnt'
]

COLS_GRADE = [col for col in COLUMNS if col not in COLS_IN_PD]
COLS_GRADE.append("target")
COLS_GRADE.remove("application_type")


with open(DATA_PATH / "columns_grade.json", "w") as f:
    json.dump(COLS_GRADE, f)

df = pd.read_csv(DATA_PATH / "loan.csv", usecols=COLUMNS)
df = df[df["application_type"] == "Individual"]
df.drop("application_type", inplace=True, axis=1)

# profile = ProfileReport(df, title="First Report")
# profile.to_file(REPORTS_PATH / "first_report.html")

df = df[df["loan_status"].isin(["Fully Paid", "Charged Off", "Default"])]
df["target"] = df["loan_status"].apply(lambda x: 1 if x in ["Charged Off", "Default"] else 0) # Maybe give it after Report

df['issue_d'] = pd.to_datetime(df['issue_d'])
df_rep = df[df["issue_d"].dt.year > 2012]

# profile = ProfileReport(df, title="Report after 2012 changes")
# profile.to_file(REPORTS_PATH / "report_after_2012.html")


train_mask = (df['issue_d'] >= '2013-01-01') & (df['issue_d'] <= '2015-12-31')
test_mask  = (df['issue_d'] >= '2016-01-01') & (df['issue_d'] <= '2016-12-31')
oot_mask   = (df['issue_d'] >= '2017-01-01') & (df['issue_d'] <= '2017-12-31')

df_train_grade = df[COLS_GRADE].copy()
df_train_grade = df_train_grade[train_mask]
df_test_grade = df[COLS_GRADE].copy()
df_test_grade = df_test_grade[test_mask]
df_oot_grade = df[COLS_GRADE].copy()
df_oot_grade = df_oot_grade[oot_mask]

df_train_grade.to_csv(DATA_PATH / "train_grade.csv", index=False)
df_test_grade.to_csv(DATA_PATH / "test_grade.csv", index=False)
df_oot_grade.to_csv(DATA_PATH / "oot_grade.csv", index=False)

df.to_csv(DATA_PATH / "destilled_dataset.csv", index=False)