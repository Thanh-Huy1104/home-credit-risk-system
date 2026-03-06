import argparse
import os
from pathlib import Path

import duckdb
import yaml


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def aggregate_bureau(con, data_raw):
    print("Aggregating bureau...")
    file_path = os.path.join(data_raw, "bureau.csv")

    con.execute(f"""
        CREATE OR REPLACE TABLE bureau_agg AS 
        SELECT 
            SK_ID_CURR,
            COUNT(SK_ID_BUREAU) AS bureau_loan_count,
            SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1 ELSE 0 END) AS bureau_active_count,
            SUM(CASE WHEN CREDIT_ACTIVE = 'Closed' THEN 1 ELSE 0 END) AS bureau_closed_count,
            SUM(AMT_CREDIT_SUM_DEBT) AS bureau_total_debt,
            SUM(AMT_CREDIT_SUM) AS bureau_total_credit,
            SUM(AMT_CREDIT_SUM_OVERDUE) AS bureau_total_overdue,
            MAX(CREDIT_DAY_OVERDUE) AS bureau_max_days_overdue,
            MAX(AMT_CREDIT_MAX_OVERDUE) AS bureau_max_credit_overdue,
            AVG(AMT_ANNUITY) AS bureau_annuity_mean,
            AVG(AMT_CREDIT_SUM) AS bureau_credit_avg,
            AVG(DAYS_CREDIT) AS bureau_days_credit_mean,
            MAX(DAYS_CREDIT) AS bureau_days_credit_max,
            SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1.0 ELSE 0 END) / COUNT(SK_ID_BUREAU) AS bureau_active_pct
        FROM read_csv_auto('{file_path}')
        GROUP BY SK_ID_CURR
    """)

    n_rows = con.execute("SELECT COUNT(*) FROM bureau_agg").fetchone()[0]
    print(f"   {n_rows:,} rows")


def aggregate_bureau_balance(con, data_raw):
    print("Aggregating bureau_balance...")
    bureau_path = os.path.join(data_raw, "bureau.csv")
    bb_path = os.path.join(data_raw, "bureau_balance.csv")

    con.execute(f"""
        CREATE OR REPLACE TABLE bureau_balance_agg AS 
        SELECT 
            b.SK_ID_CURR,
            COUNT(bb.STATUS) AS bb_count,
            SUM(CASE WHEN bb.STATUS = '0' THEN 1 ELSE 0 END) AS bb_status_0,
            SUM(CASE WHEN bb.STATUS = '1' THEN 1 ELSE 0 END) AS bb_status_1,
            SUM(CASE WHEN bb.STATUS = '2' THEN 1 ELSE 0 END) AS bb_status_2,
            SUM(CASE WHEN bb.STATUS = 'C' THEN 1 ELSE 0 END) AS bb_status_C,
            SUM(CASE WHEN bb.STATUS = 'X' THEN 1 ELSE 0 END) AS bb_status_X,
            (SUM(CASE WHEN bb.STATUS = '1' THEN 1.0 ELSE 0 END) + 
             SUM(CASE WHEN bb.STATUS = '2' THEN 1.0 ELSE 0 END)) / COUNT(bb.STATUS) AS bb_late_pct
        FROM read_csv_auto('{bureau_path}') b
        JOIN read_csv_auto('{bb_path}') bb ON b.SK_ID_BUREAU = bb.SK_ID_BUREAU
        GROUP BY b.SK_ID_CURR
    """)

    n_rows = con.execute("SELECT COUNT(*) FROM bureau_balance_agg").fetchone()[0]
    print(f"   {n_rows:,} rows")


def aggregate_previous_application(con, data_raw):
    print("Aggregating previous_application...")
    file_path = os.path.join(data_raw, "previous_application.csv")

    con.execute(f"""
        CREATE OR REPLACE TABLE prev_app_agg AS 
        SELECT 
            SK_ID_CURR,
            COUNT(SK_ID_PREV) AS prev_app_count,
            SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN 1 ELSE 0 END) AS prev_app_approved,
            SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Refused' THEN 1 ELSE 0 END) AS prev_app_refused,
            SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Canceled' THEN 1 ELSE 0 END) AS prev_app_canceled,
            AVG(AMT_APPLICATION) AS prev_app_amount_mean,
            SUM(AMT_APPLICATION) AS prev_app_amount_sum,
            AVG(AMT_CREDIT) AS prev_app_credit_mean,
            SUM(AMT_CREDIT) AS prev_app_credit_sum,
            AVG(AMT_ANNUITY) AS prev_app_annuity_mean,
            AVG(DAYS_DECISION) AS prev_app_days_decision_mean,
            MIN(DAYS_DECISION) AS prev_app_days_decision_min,
            AVG(CNT_PAYMENT) AS prev_app_cnt_payment_mean,
            SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Refused' THEN 1.0 ELSE 0 END) / COUNT(SK_ID_PREV) AS prev_app_refused_pct,
            SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN 1.0 ELSE 0 END) / COUNT(SK_ID_PREV) AS prev_app_approved_pct
        FROM read_csv_auto('{file_path}')
        GROUP BY SK_ID_CURR
    """)

    n_rows = con.execute("SELECT COUNT(*) FROM prev_app_agg").fetchone()[0]
    print(f"   {n_rows:,} rows")


def aggregate_installments_payments(con, data_raw):
    print("Aggregating installments_payments...")
    file_path = os.path.join(data_raw, "installments_payments.csv")

    con.execute(f"""
        CREATE OR REPLACE TABLE installments_agg AS 
        SELECT 
            SK_ID_CURR,
            COUNT(SK_ID_PREV) AS installments_count,
            COUNT(DISTINCT SK_ID_PREV) AS installments_num_unique,
            AVG(AMT_INSTALMENT - AMT_PAYMENT) AS payment_diff_mean,
            SUM(AMT_INSTALMENT - AMT_PAYMENT) AS payment_diff_sum,
            STDDEV(AMT_INSTALMENT - AMT_PAYMENT) AS payment_diff_std,
            AVG(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT) AS days_late_mean,
            MAX(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT) AS days_late_max,
            SUM(CASE WHEN DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT > 0 THEN 1 ELSE 0 END) AS paid_late_count,
            AVG(AMT_INSTALMENT) AS amt_instalment_mean,
            MAX(AMT_INSTALMENT) AS amt_instalment_max,
            AVG(AMT_PAYMENT) AS amt_payment_mean,
            SUM(AMT_PAYMENT) AS amt_payment_sum,
            SUM(CASE WHEN DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT > 0 THEN 1.0 ELSE 0 END) / COUNT(SK_ID_PREV) AS paid_late_pct
        FROM read_csv_auto('{file_path}')
        GROUP BY SK_ID_CURR
    """)

    n_rows = con.execute("SELECT COUNT(*) FROM installments_agg").fetchone()[0]
    print(f"   {n_rows:,} rows")


def aggregate_pos_cash(con, data_raw):
    print("Aggregating POS_CASH_balance...")
    file_path = os.path.join(data_raw, "POS_CASH_balance.csv")

    con.execute(f"""
        CREATE OR REPLACE TABLE pos_cash_agg AS 
        SELECT 
            SK_ID_CURR,
            COUNT(SK_ID_PREV) AS pos_count,
            COUNT(DISTINCT SK_ID_PREV) AS pos_num_unique,
            SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Active' THEN 1 ELSE 0 END) AS pos_status_0,
            SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Completed' THEN 1 ELSE 0 END) AS pos_status_1,
            SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Signed' THEN 1 ELSE 0 END) AS pos_status_2,
            SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Returned' THEN 1 ELSE 0 END) AS pos_status_3,
            AVG(MONTHS_BALANCE) AS pos_months_balance_min,
            AVG(MONTHS_BALANCE) AS pos_months_balance_mean,
            AVG(CNT_INSTALMENT) AS pos_cnt_instalment_mean,
            AVG(CNT_INSTALMENT_FUTURE) AS pos_cnt_instalment_future_mean
        FROM read_csv_auto('{file_path}')
        GROUP BY SK_ID_CURR
    """)

    n_rows = con.execute("SELECT COUNT(*) FROM pos_cash_agg").fetchone()[0]
    print(f"   {n_rows:,} rows")


def aggregate_credit_card(con, data_raw):
    print("Aggregating credit_card_balance...")
    file_path = os.path.join(data_raw, "credit_card_balance.csv")

    con.execute(f"""
        CREATE OR REPLACE TABLE credit_card_agg AS 
        SELECT 
            SK_ID_CURR,
            COUNT(SK_ID_PREV) AS cc_count,
            COUNT(DISTINCT SK_ID_PREV) AS cc_num_unique,
            AVG(AMT_BALANCE) AS cc_balance_mean,
            MAX(AMT_BALANCE) AS cc_balance_max,
            MIN(AMT_BALANCE) AS cc_balance_min,
            AVG(AMT_CREDIT_LIMIT_ACTUAL) AS cc_credit_limit_mean,
            AVG(AMT_DRAWINGS_ATM_CURRENT) AS cc_drawings_mean,
            SUM(AMT_DRAWINGS_ATM_CURRENT) AS cc_drawings_sum,
            AVG(AMT_PAYMENT_CURRENT) AS cc_payment_mean,
            SUM(AMT_PAYMENT_CURRENT) AS cc_payment_sum,
            AVG(CNT_INSTALMENT_MATURE_CUM) AS cc_installments_mean,
            AVG(MONTHS_BALANCE) AS cc_months_balance_mean,
            MAX(SK_DPD) AS cc_dpd_max,
            AVG(SK_DPD) AS cc_dpd_mean
        FROM read_csv_auto('{file_path}')
        GROUP BY SK_ID_CURR
    """)

    n_rows = con.execute("SELECT COUNT(*) FROM credit_card_agg").fetchone()[0]
    print(f"   {n_rows:,} rows")


def join_all_tables(con, processed_dir):
    print("Joining all tables to master...")

    con.execute("""
        CREATE OR REPLACE TABLE application_features AS
        SELECT 
            a.*,
            COALESCE(b.bureau_loan_count, 0) AS bureau_loan_count,
            COALESCE(b.bureau_active_count, 0) AS bureau_active_count,
            COALESCE(b.bureau_closed_count, 0) AS bureau_closed_count,
            COALESCE(b.bureau_total_debt, 0) AS bureau_total_debt,
            COALESCE(b.bureau_total_credit, 0) AS bureau_total_credit,
            COALESCE(b.bureau_total_overdue, 0) AS bureau_total_overdue,
            COALESCE(b.bureau_max_days_overdue, 0) AS bureau_max_days_overdue,
            COALESCE(b.bureau_max_credit_overdue, 0) AS bureau_max_credit_overdue,
            COALESCE(b.bureau_annuity_mean, 0) AS bureau_annuity_mean,
            COALESCE(b.bureau_credit_avg, 0) AS bureau_credit_avg,
            COALESCE(b.bureau_days_credit_mean, 0) AS bureau_days_credit_mean,
            COALESCE(b.bureau_days_credit_max, 0) AS bureau_days_credit_max,
            COALESCE(b.bureau_active_pct, 0) AS bureau_active_pct,
            COALESCE(bb.bb_count, 0) AS bb_count,
            COALESCE(bb.bb_status_0, 0) AS bb_status_0,
            COALESCE(bb.bb_status_1, 0) AS bb_status_1,
            COALESCE(bb.bb_status_2, 0) AS bb_status_2,
            COALESCE(bb.bb_status_C, 0) AS bb_status_C,
            COALESCE(bb.bb_status_X, 0) AS bb_status_X,
            COALESCE(bb.bb_late_pct, 0) AS bb_late_pct,
            COALESCE(pv.prev_app_count, 0) AS prev_app_count,
            COALESCE(pv.prev_app_approved, 0) AS prev_app_approved,
            COALESCE(pv.prev_app_refused, 0) AS prev_app_refused,
            COALESCE(pv.prev_app_canceled, 0) AS prev_app_canceled,
            COALESCE(pv.prev_app_amount_mean, 0) AS prev_app_amount_mean,
            COALESCE(pv.prev_app_amount_sum, 0) AS prev_app_amount_sum,
            COALESCE(pv.prev_app_credit_mean, 0) AS prev_app_credit_mean,
            COALESCE(pv.prev_app_credit_sum, 0) AS prev_app_credit_sum,
            COALESCE(pv.prev_app_annuity_mean, 0) AS prev_app_annuity_mean,
            COALESCE(pv.prev_app_days_decision_mean, 0) AS prev_app_days_decision_mean,
            COALESCE(pv.prev_app_days_decision_min, 0) AS prev_app_days_decision_min,
            COALESCE(pv.prev_app_cnt_payment_mean, 0) AS prev_app_cnt_payment_mean,
            COALESCE(pv.prev_app_refused_pct, 0) AS prev_app_refused_pct,
            COALESCE(pv.prev_app_approved_pct, 0) AS prev_app_approved_pct,
            COALESCE(ip.installments_count, 0) AS installments_count,
            COALESCE(ip.installments_num_unique, 0) AS installments_num_unique,
            COALESCE(ip.payment_diff_mean, 0) AS payment_diff_mean,
            COALESCE(ip.payment_diff_sum, 0) AS payment_diff_sum,
            COALESCE(ip.payment_diff_std, 0) AS payment_diff_std,
            COALESCE(ip.days_late_mean, 0) AS days_late_mean,
            COALESCE(ip.days_late_max, 0) AS days_late_max,
            COALESCE(ip.paid_late_count, 0) AS paid_late_count,
            COALESCE(ip.amt_instalment_mean, 0) AS amt_instalment_mean,
            COALESCE(ip.amt_instalment_max, 0) AS amt_instalment_max,
            COALESCE(ip.amt_payment_mean, 0) AS amt_payment_mean,
            COALESCE(ip.amt_payment_sum, 0) AS amt_payment_sum,
            COALESCE(ip.paid_late_pct, 0) AS paid_late_pct,
            COALESCE(pos.pos_count, 0) AS pos_count,
            COALESCE(pos.pos_num_unique, 0) AS pos_num_unique,
            COALESCE(pos.pos_status_0, 0) AS pos_status_0,
            COALESCE(pos.pos_status_1, 0) AS pos_status_1,
            COALESCE(pos.pos_months_balance_mean, 0) AS pos_months_balance_mean,
            COALESCE(pos.pos_cnt_instalment_mean, 0) AS pos_cnt_instalment_mean,
            COALESCE(pos.pos_cnt_instalment_future_mean, 0) AS pos_cnt_instalment_future_mean,
            COALESCE(cc.cc_count, 0) AS cc_count,
            COALESCE(cc.cc_num_unique, 0) AS cc_num_unique,
            COALESCE(cc.cc_balance_mean, 0) AS cc_balance_mean,
            COALESCE(cc.cc_balance_max, 0) AS cc_balance_max,
            COALESCE(cc.cc_credit_limit_mean, 0) AS cc_credit_limit_mean,
            COALESCE(cc.cc_drawings_mean, 0) AS cc_drawings_mean,
            COALESCE(cc.cc_drawings_sum, 0) AS cc_drawings_sum,
            COALESCE(cc.cc_payment_mean, 0) AS cc_payment_mean,
            COALESCE(cc.cc_payment_sum, 0) AS cc_payment_sum,
            COALESCE(cc.cc_installments_mean, 0) AS cc_installments_mean,
            COALESCE(cc.cc_dpd_max, 0) AS cc_dpd_max,
            COALESCE(cc.cc_dpd_mean, 0) AS cc_dpd_mean
        FROM application_all a
        LEFT JOIN bureau_agg b ON a.SK_ID_CURR = b.SK_ID_CURR
        LEFT JOIN bureau_balance_agg bb ON a.SK_ID_CURR = bb.SK_ID_CURR
        LEFT JOIN prev_app_agg pv ON a.SK_ID_CURR = pv.SK_ID_CURR
        LEFT JOIN installments_agg ip ON a.SK_ID_CURR = ip.SK_ID_CURR
        LEFT JOIN pos_cash_agg pos ON a.SK_ID_CURR = pos.SK_ID_CURR
        LEFT JOIN credit_card_agg cc ON a.SK_ID_CURR = cc.SK_ID_CURR
    """)

    n_rows = con.execute("SELECT COUNT(*) FROM application_features").fetchone()[0]
    n_cols = con.execute(
        "SELECT COUNT(*) FROM pragma_table_info('application_features')"
    ).fetchone()[0]
    print(f"   {n_rows:,} rows x {n_cols:,} columns in application_features")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)

    duckdb_path = cfg["paths"]["duckdb_path"]
    data_raw = cfg["paths"]["data_raw"]
    processed_dir = cfg["paths"]["data_processed"]

    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(duckdb_path)).mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(duckdb_path)

    aggregate_bureau(con, data_raw)
    aggregate_bureau_balance(con, data_raw)
    aggregate_previous_application(con, data_raw)
    aggregate_installments_payments(con, data_raw)
    aggregate_pos_cash(con, data_raw)
    aggregate_credit_card(con, data_raw)

    join_all_tables(con, processed_dir)

    print("\nWriting feature table to parquet...")
    con.execute(
        f"COPY (SELECT * FROM application_features) TO '{os.path.join(processed_dir, 'application_features.parquet')}' (FORMAT PARQUET)"
    )

    con.close()
    print("Done!")


if __name__ == "__main__":
    main()
