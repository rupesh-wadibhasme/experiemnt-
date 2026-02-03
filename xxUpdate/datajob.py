dq_config = {
    "required_columns": [],              # to be provided by you
    "expected_columns": [],              # exact column name validation
    "mandatory_non_null_columns": [],    # to be provided by you
    "null_threshold_pct": 90,             # hard fail if exceeded
    "min_row_count": 1,                   # placeholder
    "expected_dtypes": {                  # parsability check
        # "column_name": "data_type"
        # example: "trade_date": "timestamp"
    }
}
from pyspark.sql.functions import col, isnan
from pyspark.sql.types import (
    StringType, IntegerType, LongType,
    DoubleType, DecimalType, TimestampType, DateType
)

def run_data_quality_checks(df, dq_config):
    """
    Runs hard-fail data quality checks on a Spark DataFrame.

    Returns:
        dict: JSON-serializable validation results
    """

    results = {}
    overall_status = "PASS"

    total_rows = df.count()

    # -------------------------------
    # 1. Minimum Row Count Check
    # -------------------------------
    min_row_count = dq_config.get("min_row_count", 1)
    check_name = "min_row_count"

    if total_rows < min_row_count:
        results[check_name] = "FAIL"
        overall_status = "FAIL"
    else:
        results[check_name] = "PASS"

    # -------------------------------
    # 2. Required Column Presence
    # -------------------------------
    required_columns = set(dq_config.get("required_columns", []))
    actual_columns = set(df.columns)

    check_name = "required_column_presence"
    missing_columns = required_columns - actual_columns

    if missing_columns:
        results[check_name] = "FAIL"
        results["missing_required_columns"] = list(missing_columns)
        overall_status = "FAIL"
    else:
        results[check_name] = "PASS"

    # -------------------------------
    # 3. Column Name Validation (Exact Match)
    # -------------------------------
    expected_columns = set(dq_config.get("expected_columns", []))
    check_name = "column_name_validation"

    if expected_columns:
        unexpected_columns = actual_columns - expected_columns
        missing_expected_columns = expected_columns - actual_columns

        if unexpected_columns or missing_expected_columns:
            results[check_name] = "FAIL"
            results["unexpected_columns"] = list(unexpected_columns)
            results["missing_expected_columns"] = list(missing_expected_columns)
            overall_status = "FAIL"
        else:
            results[check_name] = "PASS"
    else:
        results[check_name] = "SKIPPED"

    # -------------------------------
    # 4. Parsability / Data Type Checks
    # -------------------------------
    expected_dtypes = dq_config.get("expected_dtypes", {})
    check_name = "parsability_checks"

    dtype_failures = {}

    for column, expected_type in expected_dtypes.items():
        if column not in actual_columns:
            continue

        casted_col = col(column).cast(expected_type)
        failed_cast_count = df.filter(
            col(column).isNotNull() & casted_col.isNull()
        ).count()

        if failed_cast_count > 0:
            dtype_failures[column] = failed_cast_count

    if dtype_failures:
        results[check_name] = "FAIL"
        results["datatype_cast_failures"] = dtype_failures
        overall_status = "FAIL"
    else:
        results[check_name] = "PASS"

    # -------------------------------
    # 5. Null Percentage Threshold Check
    # -------------------------------
    null_threshold = dq_config.get("null_threshold_pct", 90)
    check_name = "null_percentage_threshold"

    null_failures = {}

    for column in actual_columns:
        null_count = df.filter(
            col(column).isNull() | isnan(col(column))
        ).count()

        null_pct = (null_count / total_rows) * 100 if total_rows > 0 else 100

        if null_pct >= null_threshold:
            null_failures[column] = round(null_pct, 2)

    if null_failures:
        results[check_name] = "FAIL"
        results["null_percentage_exceeded"] = null_failures
        overall_status = "FAIL"
    else:
        results[check_name] = "PASS"

    # -------------------------------
    # 6. Mandatory Non-Null Columns
    # -------------------------------
    mandatory_non_null_columns = dq_config.get(
        "mandatory_non_null_columns", []
    )

    check_name = "mandatory_non_null_columns"

    non_null_failures = {}

    for column in mandatory_non_null_columns:
        if column not in actual_columns:
            non_null_failures[column] = "COLUMN_MISSING"
            continue

        null_count = df.filter(col(column).isNull()).count()
        if null_count > 0:
            non_null_failures[column] = null_count

    if non_null_failures:
        results[check_name] = "FAIL"
        results["mandatory_null_violations"] = non_null_failures
        overall_status = "FAIL"
    else:
        results[check_name] = "PASS"

    # -------------------------------
    # Final Result
    # -------------------------------
    return {
        "overall_status": overall_status,
        "row_count": total_rows,
        "validations": results
    }
