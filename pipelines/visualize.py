import duckdb


def main():
    con = duckdb.connect("data/duckdb/home_credit.duckdb")

    print("=" * 60)
    print("HOME CREDIT RISK - DATA EXPLORATION")
    print("=" * 60)

    print("\n### TABLES IN DATABASE ###")
    tables = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchall()
    for t in tables:
        count = con.execute(f"SELECT COUNT(*) FROM {t[0]}").fetchone()[0]
        print(f"  {t[0]:<30} {count:>12,} rows")

    print("\n### APPLICATION_FEATURES OVERVIEW ###")
    row_count = con.execute("SELECT COUNT(*) FROM application_features").fetchone()[0]
    col_count = con.execute(
        "SELECT COUNT(*) FROM pragma_table_info('application_features')"
    ).fetchone()[0]
    print(f"  Total rows: {row_count:,}")
    print(f"  Total columns: {col_count}")

    print("\n### TRAIN vs TEST SPLIT ###")
    split = con.execute("""
        SELECT 
            is_train,
            CASE WHEN is_train = 1 THEN 'TRAIN' ELSE 'TEST' END as dataset,
            COUNT(*) as count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as pct
        FROM application_features
        GROUP BY is_train
        ORDER BY is_train DESC
    """).fetchall()
    for _is_train, dataset, count, pct in split:
        print(f"  {dataset:<8} {count:>10,} rows  ({pct:>5}%)")

    print("\n### TARGET DISTRIBUTION (TRAIN ONLY) ###")
    target = con.execute("""
        SELECT 
            TARGET,
            CASE TARGET WHEN 0 THEN 'No Default' ELSE 'Default' END as label,
            COUNT(*) as count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as pct
        FROM application_features
        WHERE is_train = 1
        GROUP BY TARGET
        ORDER BY TARGET
    """).fetchall()
    for target_val, label, count, pct in target:
        print(f"  TARGET={target_val} ({label:<11}) {count:>10,} rows  ({pct:>5}%)")

    imbalance = con.execute("""
        SELECT ROUND(1.0 * SUM(CASE WHEN TARGET = 0 THEN 1 ELSE 0 END) * 1.0 / SUM(CASE WHEN TARGET = 1 THEN 1 ELSE 0 END), 2)
        FROM application_features
        WHERE is_train = 1
    """).fetchone()[0]
    print(f"  Imbalance ratio (no default : default): {imbalance:.1f} : 1")

    print("\n### COLUMN CATEGORIES ###")

    base_cols = ["SK_ID_CURR", "TARGET", "is_train"]
    bureau_cols = [
        c
        for c in con.execute(
            "SELECT name FROM pragma_table_info('application_features') WHERE name LIKE 'bureau_%' OR name LIKE 'bb_%'"
        ).fetchall()
    ]
    prev_cols = [
        c
        for c in con.execute(
            "SELECT name FROM pragma_table_info('application_features') WHERE name LIKE 'prev_%'"
        ).fetchall()
    ]
    install_cols = [
        c
        for c in con.execute(
            "SELECT name FROM pragma_table_info('application_features') WHERE name LIKE 'installments_%' OR name LIKE 'payment_%' OR name LIKE 'paid_%' OR name LIKE 'days_late%' OR name LIKE 'amt_%'"
        ).fetchall()
    ]
    pos_cols = [
        c
        for c in con.execute(
            "SELECT name FROM pragma_table_info('application_features') WHERE name LIKE 'pos_%'"
        ).fetchall()
    ]
    cc_cols = [
        c
        for c in con.execute(
            "SELECT name FROM pragma_table_info('application_features') WHERE name LIKE 'cc_%'"
        ).fetchall()
    ]

    print(f"  Base columns (ID, target): {len(base_cols)}")
    print(f"  Bureau (external credit history): {len(bureau_cols)}")
    print(f"  Previous application (internal): {len(prev_cols)}")
    print(f"  Installments/payments: {len(install_cols)}")
    print(f"  POS cash: {len(pos_cols)}")
    print(f"  Credit card: {len(cc_cols)}")

    print("\n### FEATURE GROUPS (sample) ###")
    print("  Bureau examples:", [c[0] for c in bureau_cols[:5]])
    print("  Previous app examples:", [c[0] for c in prev_cols[:5]])
    print("  Installments examples:", [c[0] for c in install_cols[:5]])

    print("\n### KEY AGGREGATED FEATURES ###")
    stats = con.execute("""
        SELECT 
            'bureau_loan_count' as feature,
            MIN(bureau_loan_count) as min,
            MAX(bureau_loan_count) as max,
            ROUND(AVG(bureau_loan_count), 2) as mean,
            ROUND(MEDIAN(bureau_loan_count), 2) as median
        FROM application_features
        WHERE bureau_loan_count > 0
        UNION ALL
        SELECT 
            'prev_app_refused' as feature,
            MIN(prev_app_refused) as min,
            MAX(prev_app_refused) as max,
            ROUND(AVG(prev_app_refused), 2) as mean,
            ROUND(MEDIAN(prev_app_refused), 2) as median
        FROM application_features
        WHERE prev_app_refused > 0
        UNION ALL
        SELECT 
            'paid_late_pct' as feature,
            MIN(paid_late_pct) as min,
            MAX(paid_late_pct) as max,
            ROUND(AVG(paid_late_pct), 4) as mean,
            ROUND(MEDIAN(paid_late_pct), 4) as median
        FROM application_features
        WHERE paid_late_pct > 0
    """).fetchall()
    print(f"  {'Feature':<20} {'Min':>8} {'Max':>8} {'Mean':>10} {'Median':>10}")
    print("  " + "-" * 58)
    for feature, min_val, max_val, mean_val, median_val in stats:
        print(f"  {feature:<20} {min_val:>8} {max_val:>8} {mean_val:>10} {median_val:>10}")

    print("\n### CREDIT HISTORY SUMMARY ###")
    credit = con.execute("""
        SELECT 
            COUNT(DISTINCT SK_ID_CURR) as unique_clients,
            SUM(bureau_loan_count) as total_bureau_loans,
            SUM(prev_app_count) as total_prev_applications,
            SUM(installments_count) as total_installment_payments,
            ROUND(AVG(bureau_total_debt), 2) as avg_total_debt,
            ROUND(AVG(bureau_active_pct), 4) as avg_active_pct
        FROM application_features
    """).fetchone()
    print(f"  Unique clients: {credit[0]:,}")
    print(f"  Total bureau loans: {credit[1]:,.0f}")
    print(f"  Total previous applications: {credit[2]:,.0f}")
    print(f"  Total installment payments: {credit[3]:,.0f}")
    print(f"  Avg total debt: ${credit[4]:,.2f}")
    print(f"  Avg active loan %: {credit[5] * 100:.1f}%")

    print("\n### RISK INDICATORS (TRAIN) ###")
    risk = con.execute("""
        SELECT 
            ROUND(AVG(bureau_active_pct), 4) as avg_active_pct,
            ROUND(AVG(prev_app_refused_pct), 4) as avg_refused_pct,
            ROUND(AVG(paid_late_pct), 4) as avg_late_pct,
            ROUND(AVG(CASE WHEN bureau_max_days_overdue > 0 THEN 1.0 ELSE 0.0 END), 4) as pct_with_overdue
        FROM application_features
        WHERE is_train = 1
    """).fetchone()
    print(f"  Avg active bureau loan %: {risk[0] * 100:.1f}%")
    print(f"  Avg refused application %: {risk[1] * 100:.1f}%")
    print(f"  Avg late payment %: {risk[2] * 100:.1f}%")
    print(f"  % clients with overdue: {risk[3] * 100:.1f}%")

    print("\n" + "=" * 60)
    print("EXPLANATION: TEST vs TRAIN")
    print("=" * 60)
    print("""
The application_features table contains both TRAIN and TEST data:

  - TRAIN (is_train=1): Has TARGET values (0=no default, 1=default)
    Used for model training. The model learns to predict default.

  - TEST (is_train=0): TARGET is NULL
    These are new applicants. The model predicts their TARGET.

  - is_train column: Flags which split each row belongs to.
    1 = training data (has labels)
    0 = test data (labels hidden, to be predicted)

The is_train column was added during ingestion to keep both
sets in one table for feature engineering, then split again
for training/prediction.
""")

    con.close()


if __name__ == "__main__":
    main()
