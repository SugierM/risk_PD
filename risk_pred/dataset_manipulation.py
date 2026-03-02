######## TO LEAVE
# "loanAmnt", "term", "installment", "purpose", "annualInc", "application_type", "empLength"
# "emp_title", "homeOwnership", "dti", "delinq2Yrs", "mthsSinceLastDelinq", "pubRec", "inq_last_12m",
# "inqLast6Mths, "openAcc", "totalAcc", "revolBal", "revolUtil", "accNowDelinq", "tot_cur_bal",
# "mortAcc", "earliestCrLine", "avg_cur_bal", "num_tl_op_past_12m", "pct_tl_nvr_dlq",
# "ficoRangeHigh", "ficoRangeLow", "chargeoff_within_12_mths", "mo_sin_old_rev_tl", "inqLast6Mths",
# "bcUtil", "num_actv_bc_tl", "num_tl_90g_dpd_24m", "tot_coll_amt", "total_rev_hi_lim", "initial_list_status",
# "inq_fi", "inq_last_12m", 

import pandas as pd
import os

COLUMNS = ["loan_amnt", "term", "installment", "purpose", "annual_inc", "application_type", 
        "emp_length", "emp_title", "home_ownership", "dti", "delinq_2yrs", "mths_since_last_delinq", "pub_rec",
        "inq_last_6mths", "open_acc", "total_acc", "revol_bal", "revol_util", "acc_now_delinq", "tot_cur_bal",
        "mort_acc", "earliest_cr_line", "avg_cur_bal", "num_tl_op_past_12m", "pct_tl_nvr_dlq",
        "chargeoff_within_12_mths", "mo_sin_old_rev_tl_op", # "fico_range_high", "fico_range_low" - are not present
        "bc_util", "num_actv_bc_tl", "num_tl_90g_dpd_24m", "tot_coll_amt", "total_rev_hi_lim", "initial_list_status",
        "inq_fi", "inq_last_12m", "loan_status", "issue_d"
]



def view_columns(name) -> None:
    """
    Docstring for view_columns
    
    :param name: Description
    """

    real_headers = pd.read_csv(f"../data/{name}", nrows=0, encoding='utf-8-sig').columns.to_list()

    missing = [c for c in COLUMNS if c not in real_headers]
    print(f"________________________ {name} _______________________\n\n\n")
    
    print("________________________ MISSING _______________________")
    print(missing)
    print("________________________________________________________")

    print("_____________________ REAL HEADERS _____________________")
    for i in range(0, len(real_headers), 3):
        print(real_headers[i:min(i+3, len(real_headers))])
    print("________________________________________________________\n\n\n")


def create_dataset() -> None:
    """
    Docstring for create_dataset
    """
    if not os.path.exists('data'):
        os.makedirs('data')

    df = pd.read_csv("../data/loan.csv", usecols=COLUMNS)
    df = df[df["application_type"] == "Individual"]
    df.to_csv("data/destilled_dataset.csv", index=False)


def check_data() -> None:
    """
    Docstring for check_data
    """

    df = pd.read_csv("../data/loan.csv", usecols=['issue_d', 'hardship_start_date'])
    print(df.info())
    df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y')
    df['hardship_start_date'] = pd.to_datetime(df['hardship_start_date'], format='%b-%Y')

    print("Issued")
    print(*list(df["issue_d"].sort_values().unique()), sep="\n")
    print("_________________________________")

    
    print(df.info())
    print(df.head())
    print(df["hardship_start_date"].unique())
    # df_hardship = df.dropna(subset=['hardship_start_date']).copy()

    # df_hardship['is_before_issue'] = df_hardship['hardship_start_date'] < df_hardship['issue_d']

    # before_count = df_hardship['is_before_issue'].sum()
    # total_hardship = len(df_hardship)

    # print(f"Liczba sprawdzonych rekordów z hardship: {total_hardship}")
    # print(f"Liczba przypadków, gdzie hardship zaczął się PRZED udzieleniem kredytu: {before_count}")

    # if before_count > 0:
    #     print("\nPrzykładowe rekordy 'sprzed' udzielenia:")
    #     print(df_hardship[df_hardship['is_before_issue'] == True].head())
    # else:
    #     print("\nWe wszystkich przypadkach hardship zaczął się po udzieleniu kredytu (lub w tym samym miesiącu).") 



def lean_dataset() -> None:
    """
    Docstring for lean_dataset
    """
    df = pd.read_csv("data/destilled_dataset.csv")
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off", "Default"])]

    df["target"] = df["loan_status"].apply(lambda x: 1 if x in ["Charged Off", "Default"] else 0)

    df_len = len(df)
    print(df_len)
    print(f"% Fully Paid = {(len(df[df["loan_status"] == "Fully Paid"]) / df_len) * 100:.2f} %")

    print(df.head())
    print("_____________" * 3)
    print(df["target"].unique().tolist())

    df.to_csv("data/lean_dataset.csv", index=False)


    # Installment - high corr | Leave loan amount ... and term if needed
    # Emp title maybe skip for now as other models or exprests should look into it and data can be counter-intuitive in dynamic fields
    # emp_length maybe change into dummies or something
    # home_ownership - may not work with more people that rent will be present
    # CHECK annual inc and dti 
    # DTI has -1?
    # Leave delinq_2yrs and get rid off - num_tl_90g_dpd_24m
    # earliest cr line i think is fine
    # open_acc / num_actv_bc_tl /  total_acc / total_rev_hi_lim ---- total_rev_hi_lim can be good
    # revol_util / bc_util
    # initial_list_status - banks cannot do Fractional loans, but inquiry can still happen
    # acc_now_delinq, tot_coll_amount
    # tot_cur_bal / avg_cur_bal / mort_acc
    # inq_fi is fine
    # chargeoff_within_12_mths
    # mo_sin_old_rev_tl_op
    # num_tl_op_past_12m / inq_last_12m
    # pct_tl_nvr_dlq


def first_raport(filename="destilled_dataset") -> None:
    """
    Docstring for first_raport
    """

    from ydata_profiling import ProfileReport

    df = pd.read_csv(f'data/{filename}.csv')
    profile = ProfileReport(df, title="First Report")
    profile.to_file("data/first_report.html")
    print("Done")


if __name__ == "__main__":
    # create_dataset()
    # lean_dataset()
    first_raport("ready_train_dataset")