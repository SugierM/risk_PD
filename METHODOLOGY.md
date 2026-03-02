# Methodological Documentation of the PD Model

This document describes the assumptions and analytical decisions made during the development of the Probability of Default (PD) model based on LendingClub data.

---

## 1. Target Variable Definition

For the purpose of this project, the following definition of default was adopted:

*   **Bad (Default = 1):** Loans with statuses indicating capital loss or severe delinquency:
    *   `Charged Off` 
    *   `Default` 

*   **Good (Default = 0):** Loans that were properly serviced:
    *   `Fully Paid`


Undefined statuses (`In Grace Period`, `Current`) were excluded from the analysis due to the lack of information about the final repayment outcome.
Additionally, in line with the standard definition of default in Poland (90+ days past due), observations with delinquencies in the 30–120 days range were excluded due to the inability to isolate precisely the 90+ days segment.

---

## 2. Data Scope and Exclusions

To ensure the model’s realism and robustness, additional filters were applied:

*   **Decision:** All loans issued before 2013 were excluded.
*   **Justification:** This decision was driven by a significant shift in borrower profile and data quality. The pre-2013 dataset contained a substantial proportion of missing values. Furthermore, according to external sources, after 2012 LendingClub tightened its credit policy, rejecting approximately 90% of applications and introducing a minimum FICO score requirement of 660 points.
Source: [Wikipedia](https://en.wikipedia.org/wiki/LendingClub)

### 2.2. Prevention of Data Leakage
*   **Decision:** All behavioral variables generated after loan origination were removed (e.g., recovery-related variables and  `hardship` status indicators).
*   **Justification:** Variables such as `recoveries` or `total_rec_prncp` become known only after a credit event has occurred. Including them in a PD model (which is constructed at the loan origination stage) would introduce data leakage and artificially inflate model performance.

*   **Decision:** The `grade` variable was excluded from the model development process.
*   **Justification:** Including it would effectively replicate LendingClub’s internal scoring methodology rather than develop an independent risk model. Additionally its inclusion could introduce circularity and reduce model transparency. Therefore, it was excluded to ensure methodological independence and governance clarity.

---

## 3. Data Dictionary

The dataset did not include official column descriptions; therefore, variable definitions were compiled based on external sources and with the assistance of the Gemini 3 Flash model.
The table below includes only variables classified as “available at application” — meaning variables that would realistically be accessible to a bank at the time of credit assessment.
At an early stage of the analysis, several variables were intentionally excluded. For example, geolocation data such as zip_code was removed for regulatory and ethical reasons (potential discrimination risk).

| Nazwa zmiennej | Opis danych |
| :--- | :--- |
loan_amnt | Loan amount requested by the borrower |
term | Loan repayment term (36 or 60 months) |
installment | Monthly installment amount |
emp_title | Job title provided by the borrower |
emp_length | Employment length (in years) |
home_ownership | Home ownership status |
annual_inc | Borrower’s declared annual gross income |
loan_status | Current loan status (e.g., fully paid, in repayment, delinquent, charged off) |
purpose | Loan purpose declared by the borrower |
dti | Debt-to-Income ratio – monthly obligations divided by monthly income (excluding the requested loan) |
delinq_2yrs | Number of delinquencies (30+ DPD) in the past 2 years |
earliest_cr_line | Date of borrower’s first credit line |
inq_last_6mths | Number of credit inquiries in the past 6 months |
mths_since_last_delinq | Months since last delinquency |
open_acc | Number of open credit accounts |
pub_rec | Number of derogatory public records (e.g., bankruptcies, legal judgments) |
revol_bal | Total balance on revolving credit lines |
revol_util | Revolving credit utilization ratio |
total_acc | Total number of credit lines (open and closed) |
initial_list_status | Initial listing status on the platform (W – Whole / F – Fractional) |
application_type | Application type (only “Individual”) |
acc_now_delinq | Number of accounts currently delinquent |
tot_coll_amt | Total amount ever in collections |
tot_cur_bal | Total current balance across all credit accounts |
total_rev_hi_lim | Total revolving credit limit |
inq_fi | Number of credit inquiries at financial institutions |
inq_last_12m | Total number of credit inquiries in the past 12 months |
avg_cur_bal | Average balance across active accounts |
bc_util | Bankcard utilization ratio |
chargeoff_within_12_mths | Number of charge-offs within the past 12 months |
mo_sin_old_rev_tl_op | Months since oldest revolving account was opened |
mort_acc | Number of mortgage accounts |
num_actv_bc_tl | Number of currently active credit card accounts |
num_tl_90g_dpd_24m | Number of accounts with 90+ DPD in the past 24 months |
num_tl_op_past_12m | Number of accounts opened in the past 12 months |
pct_tl_nvr_dlq | Percentage of credit lines that have never been delinquent |

---

## 4. Simulation of External Bureau Data (Proxy BIK / FICO)

The original dataset did not contain raw FICO scores. To better align the project with European banking practice, a synthetic variable imitating an external credit bureau score was constructed.

*   **Approach:** A group of historical variables (not used in the main PD model) was used to train neural network. The output of this model serves as a synthetic external credit score.
*   **Justification:** PD models used in banking decision processes almost always incorporate data from external credit bureaus (e.g., BIK in Poland). Introducing a synthetic bureau-like score increases the realism of the modeling framework.

---

## 5. Population Stability Analysis (PSI)

To ensure that the model remains stable over time, a **Population Stability Index (PSI)**. analysis was conducted. The analysis revealed moderate distributional shifts (0.1 < PSI < 0.2) for three key variables.

1.  **loan_amnt:** 
    *   *Distribution:* A higher share of smaller loans was observed in later periods. Additionally, loan limits were increased after 2016. Source: [debanked.com](https://debanked.com/2016/03/lending-club-loan-size-cap-raised-to-40000-should-investors-be-worried/#:~:text=Lending%20Club%20Loan%20Size%20Cap%20Raised%20to,Investors%20Be%20Worried?).
    *   *PSI:* 0.1467.
2.  **bc_util:** 
    *   *Distribution:* The borrower population appears to have become more conservative, or customers may have been granted higher limits that are not fully utilized.
    *   *PSI:* 0.1421
3.  **revol_util:** 
    *   *Distribution:* Borrowers in the newer sample exhibit lower utilization of revolving credit lines.
    *   *PSI:* 0.1459

For `bc_util` and `revol_util` values below 0.2 should not materially impact model performance, but they should be monitored.
Despite the lack of conclusive data confirming the underlying cause, the shift may be partially related to the 2016 governance scandal, after which LendingClub reportedly focused on higher-quality (“safer”) borrowers.
"In December 2017, the Financial Times reported that LendingClub "has struggled to overcome the effects of a governance scandal last May", and that the firm "has battled to keep big investors buying loans" despite improvements to its internal governance" [Wikipedia](https://en.wikipedia.org/wiki/LendingClub#Scandal_and_struggle,_2016-2017)

(*Interpretation note* “last May” is interpreted as referring to May 2016, when the CEO resigned and the company revised its internal policies.)

---
