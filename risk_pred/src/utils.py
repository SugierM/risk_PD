import pandas as pd
import numpy as np


COLS_IN_PD = ['annual_inc', 'target', 'num_tl_op_past_12m', 'inq_last_6mths',
 'home_ownership', 'tot_cur_bal', 'total_rev_hi_lim', 'bc_util', 'dti',
 'mo_sin_old_rev_tl_op', 'revol_util', 'loan_amnt', 'issue_d'
]

BC_UTIL_MEDIAN = 67.4
REVOL_UTIL_MEDIAN = 55.4

MAPPED_FILLS = {
    "revol_util": REVOL_UTIL_MEDIAN,
    "bc_util": BC_UTIL_MEDIAN,
    "total_rev_hi_lim": 0,
    "inq_last_6mths": 0,
    "annual_inc": 0,
    "home_ownership": "",
    "num_tl_op_past_12m": 0,
    "probability": 0,
    "bc_util": 0,
    "dti": 0,
    "target": 0,
    "loan_amnt": 0,
    "revol_util": 0,
    "tot_cur_bal": 0,
    "mo_sin_old_rev_tl_op": 0,
}


MAPPED_BINS = {
    "loan_amnt": {
        "bins": [-np.inf, 7000.0, 10500.0, 15000.0, 21000.0, np.inf],
        "labels": ["0-7k", "7k-10.5k", "10.5k-15k", "15k-21k", "21k+"]
    },
    "bc_util": {
        "bins": [-np.inf, 33.8, 52.6, 67.4, 80.6, 92.3, np.inf],
        "labels": ["0-33.8", "33.8-52.6", "52.6-67.4", "67.4-80.6", "80.6-92.3", "92.3+"]
    },
    "revol_util": {
        "bins": [-np.inf, 25.8, 37.5, 46.9, 55.4, 63.8, 72.8, 83.6, np.inf],
        "labels": ["0-25.8", "25.8-37.5", "37.5-46.9", "46.9-55.4", "55.4-63.8", "63.8-72.8", "72.8-83.6", "83.6+"]
    },
    "mo_sin_old_rev_tl_op": {
        "bins": [-np.inf, 96.0, 124.0, 156.0, 216.0, np.inf],
        "labels": ["0-96", "96-124", "124-156", "156-216", "216+"]
    },
    "num_tl_op_past_12m": {
        "bins": [-np.inf, 1.0, 2.0, 3.0, np.inf],
        "labels": ["0-1", "1-2", "2-3", "3+"]
    },
    "dti": {
        "bins": [-np.inf, 8.52, 12.06, 14.97, 17.8, 20.76, 24.2, 28.61, np.inf],
        "labels": ["0-8.5", "8.5-12.1", "12.1-15", "15-17.8", "17.8-20.8", "20.8-24.2", "24.2-28.6", "28.6+"]
    },
    "annual_inc": {
        "bins": [-np.inf, 40000.0, 51000.0, 65000.0, 80000.0, 105000.0, np.inf],
        "labels": ["0-40k", "40k-51k", "51k-65k", "65k-80k", "80k-105k", "105k+"]
    },
    "tot_cur_bal": {
        "bins": [-np.inf, 225000.0, np.inf],
        "labels": ["0-225k", "225+"]
    },
    "total_rev_hi_lim": {
        "bins": [-np.inf, 9200.0, 13700.0, 18200.0, 23300.0, 29900.0, 39232.5, 56000.0, np.inf],
        "labels": ["0-9.2k", "9.2k-13.7k", "13.7k-18.2k", "18.2k-23.3k", "23.3k-29.9k", "29.9k-39.2k", "39.2k-56k", "56k+"]
    },
    "inq_last_6mths": {
        "bins": [-np.inf, 0.001, 1, 3, np.inf],
        "labels": ["0", "1", "2", "3+"]
    },
    "probability": {
        "bins": [-np.inf, 0.341, 0.404, 0.508, 0.540, np.inf],
        "labels": ["0-0.341", "0.341-0.404", "0.404-0.508", "0.508-0.540", "0.540+"]
    }
}


WOE_MAPS = {'loan_amnt': {'0-7k': -0.3096895566140675,
  '7k-10.5k': -0.14839396489719112,
  '10.5k-15k': 0.07317771133955554,
  '15k-21k': 0.16060278332058156,
  '21k+': 0.17602849687365563},
 'bc_util': {'0-33.8': -0.23579124561341921,
  '33.8-52.6': -0.13663951809007416,
  '52.6-67.4': -0.026369763568697156,
  '67.4-80.6': 0.03168179117677081,
  '80.6-92.3': 0.08826967992070858,
  '92.3+': 0.24881094746796417},
 'revol_util': {'0-25.8': -0.2824252604509185,
  '25.8-37.5': -0.1409502448649134,
  '37.5-46.9': -0.05808321322576112,
  '46.9-55.4': 0.015557561632685266,
  '55.4-63.8': 0.04599929351808013,
  '63.8-72.8': 0.09121901668652292,
  '72.8-83.6': 0.12872654443961756,
  '83.6+': 0.15432806990206338},
 'mo_sin_old_rev_tl_op': {'0-96': 0.23358477028086658,
  '96-124': 0.12032730093963011,
  '124-156': 0.036942364961302016,
  '156-216': -0.04709502544681096,
  '216+': -0.17260901611764895},
 'num_tl_op_past_12m': {'0-1': -0.2542927947281371,
  '1-2': -0.0007594862176777524,
  '2-3': 0.1638083404163349,
  '3+': 0.37094490743331643},
 'dti': {'0-8.5': -0.41664729698886355,
  '8.5-12.1': -0.3061746434156198,
  '12.1-15': -0.2038469971829124,
  '15-17.8': -0.08790491369975961,
  '17.8-20.8': 0.0011304469747104092,
  '20.8-24.2': 0.1189383844451924,
  '24.2-28.6': 0.24248790317277355,
  '28.6+': 0.465736830450999},
 'annual_inc': {'0-40k': 0.19963982449243603,
  '40k-51k': 0.13655080027576733,
  '51k-65k': 0.07973682627196681,
  '65k-80k': -0.02948139995187071,
  '80k-105k': -0.1603270851803099,
  '105k+': -0.33379211226936023},
 'tot_cur_bal': {'0-225k': 0.07951257627272983, '225+': -0.3216473412888831},
 'total_rev_hi_lim': {'0-9.2k': 0.12877358914775885,
  '9.2k-13.7k': 0.11002361996842536,
  '13.7k-18.2k': 0.09622995436438846,
  '18.2k-23.3k': 0.08376760572862255,
  '23.3k-29.9k': 0.05410093792673,
  '29.9k-39.2k': -0.008417323992240485,
  '39.2k-56k': -0.13304169795024812,
  '56k+': -0.40254764610023425},
 'inq_last_6mths': {'0': -0.1398285168747474,
  '1': 0.08492972354556345,
  '2': 0.27875993714116065,
  '3+': 0.40302907127921617},
 'probability': {'0-0.341': -0.3821256113082504,
  '0.341-0.404': 0.39033261390951257,
  '0.404-0.508': 0.4660069085519327,
  '0.508-0.540': 0.5966661554896383,
  '0.540+': 0.9957668457803415},
 'home_ownership': {'MORTGAGE': -0.15704912558200876,
  'OWN_OTHER': 0.029727634071044905,
  'RENT': 0.17035630554360756}}


def prepare_df(df, feature_order):
    df['term_60m'] = df['term'].str.replace(" months", "").astype(float) # risky, but should work fine
    df['term_60m'] = (df['term_60m'] == 60).astype(int)
    df = df.drop(columns=['term'])


    df['emp_length_is_null'] = df['emp_length'].isnull().astype(int)
    df['emp_length_num'] = df['emp_length'].str.extract('(\d+)').astype(float)
    df['emp_length_num'] = df['emp_length_num'].fillna(-1)


    df['inq_fi'] = df['inq_fi'].fillna(-1)
    df['inq_last_12m'] = df['inq_last_12m'].fillna(-1)
    df['fi_to_total_ratio'] = np.where(
        df['inq_last_12m'] > 0, 
        df['inq_fi'] / df['inq_last_12m'], 
        0
    ) 


    df['is_never_delinq'] = df['mths_since_last_delinq'].isnull().astype(int)
    df['mths_since_last_delinq'] = df['mths_since_last_delinq'].fillna(-1)


    df['issue_d'] = pd.to_datetime(df['issue_d'])
    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'])
    df['credit_hist_months'] = (
        (df['issue_d'].dt.year - df['earliest_cr_line'].dt.year) * 12 +
        (df['issue_d'].dt.month - df['earliest_cr_line'].dt.month)
    )


    df['credit_hist_months'] = df['credit_hist_months'].apply(lambda x: max(x, 0))
    df['credit_hist_months'] = df['credit_hist_months'].fillna(-1) # There aren't any NaN values anyway


    cols_to_transform = ["revol_bal", "avg_cur_bal"]
    for col in cols_to_transform:
        upper_limit = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=upper_limit)
        df[col] = np.log10(df[col] + 1)


    df['inq_recent_ratio'] = df['inq_last_6mths'] / (df['inq_last_12m'] + 0.001)
    df['utilization_proxy'] = df['revol_bal'] / (df['avg_cur_bal'] + 0.001)
    

    cat_cols = ['purpose']
    df = pd.get_dummies(df, columns=cat_cols)
    df = df.reindex(columns=feature_order, fill_value=0)

    df = df.fillna(0)
    return df




def home_owner(val):
    if val in ["RENT", "MORTGAGE"]:
        return val
    return "OWN_OTHER"


def apply_bins(df, bin_map, mapped_fills):
    df_mapped = df.copy()
    
    df_mapped = df_mapped.fillna(mapped_fills)
    for col, config in bin_map.items():
        if col in df_mapped.columns:
            df_mapped[f"{col}_bins"] = pd.cut(
                df_mapped[col], 
                bins=config['bins'], 
                labels=config['labels'], 
                include_lowest=True
            ).astype(str)
    return df_mapped


def preprocess_data(df, feature_order=None, revol_99=None, avg_cur_99=None):
    if not revol_99 and avg_cur_99:
        raise ValueError("You need to provide 99th quantile from training dataset.")
    
    df = df.copy()

    df['term_60m'] = df['term'].str.extract(r'(\d+)').astype(float)
    df['term_60m'] = (df['term_60m'] == 60).astype(int)
    
    df['emp_length_is_null'] = df['emp_length'].isnull().astype(int)
    df['emp_length_num'] = df['emp_length'].str.extract(r'(\d+)').astype(float).fillna(-1)

    df['inq_fi'] = df['inq_fi'].fillna(-1)
    df['inq_last_12m'] = df['inq_last_12m'].fillna(-1)
    df['fi_to_total_ratio'] = np.where(df['inq_last_12m'] > 0, df['inq_fi'] / df['inq_last_12m'], 0)

    df['is_never_delinq'] = df['mths_since_last_delinq'].isnull().astype(int)
    df['mths_since_last_delinq'] = df['mths_since_last_delinq'].fillna(-1)

    df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y', errors='coerce')
    df['credit_hist_months'] = ((df['issue_d'].dt.year - df['earliest_cr_line'].dt.year) * 12 + 
                                (df['issue_d'].dt.month - df['earliest_cr_line'].dt.month)).fillna(-1)
    df['credit_hist_months'] = df['credit_hist_months'].clip(lower=0)

    df["revol_bal"] = df["revol_bal"].clip(upper=revol_99)
    df["revol_bal"] = np.log10(df["revol_bal"] + 1)
    df["avg_cur_bal"] = df["avg_cur_bal"].clip(upper=avg_cur_99)
    df["avg_cur_bal"] = np.log10(df["avg_cur_bal"] + 1)

    df['inq_recent_ratio'] = df['inq_last_6mths'] / (df['inq_last_12m'] + 0.001)
    df['utilization_proxy'] = df['revol_bal'] / (df['avg_cur_bal'] + 0.001)

    cat_cols = ['purpose']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    if feature_order:
        df = df.reindex(columns=feature_order, fill_value=0)
    
    to_drop = [ 'term', 'application_type', 'emp_title', 'issue_d', 'loan_status', 'initial_list_status', 
               'earliest_cr_line', 'emp_length', 'earliest_cr_line'] 
    df = df.drop(columns=[c for c in to_drop if c in df.columns], errors='ignore')
    
    return df.fillna(0)



def prepare_for_pd(df_raw, assets):
    df = df_raw.copy()
    mapped_bins = assets["mapped_bins"]
    woe_maps = assets["woe_maps"]
    mapped_fills = assets["mapped_fills"]
    feature_order = assets["pd_order"]

    for col, value in mapped_fills.items():
        if col in df.columns:
            df[col] = df[col].fillna(value)

    df["home_ownership"] = df["home_ownership"].astype(str).apply(home_owner)

    for col, config in mapped_bins.items():
        bin_col = col + "_bins"

        if config["bins"] is not None:
            df[bin_col] = pd.cut(
                df[col],
                bins=config["bins"],
                labels=config["labels"],
                include_lowest=True,
            )
        else:

            df[bin_col] = (
                df[col]
                .astype(str)
                .apply(home_owner)
            )
        df[bin_col] = (
            df[bin_col]
            .map(woe_maps[col])
            .astype(float)
            .fillna(0.0)
        )

    X = df.reindex(columns=feature_order, fill_value=0)

    X = X.fillna(0)

    return X


def apply_woe_transformation(df, mapped_bins, woe_maps):
    df_new = df.copy()

    for col, config in mapped_bins.items():
        required_cols = list(mapped_bins.keys())

        missing_cols = [
            col for col in required_cols
            if col not in df_new.columns
        ]

        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}"
            )

        bin_col = col + "_bins"

        if config["bins"] is not None:
            df_new[bin_col] = pd.cut(
                df_new[col],
                bins=config["bins"],
                labels=config["labels"],
                include_lowest=True,
            )

        else:
            df_new[bin_col] = (
                df_new[col]
                .astype(str)
                .apply(home_owner)
            )

        df_new[bin_col] = df_new[bin_col].map(woe_maps[col]).astype(float).fillna(0.0)


    return df_new



def home_owner(val):
    if val in ["RENT", "MORTGAGE"]:
        return val
    return "OWN_OTHER"