# =============================================================================
# MainFitbit.py
# CSE-012 Course Project — Data Theme: Heart Rate
#
# Entry point for the project. Imports all functions from FitbitFunctions.py
# and runs Tasks 1, 2, and 3 in order across both data files:
#
#   2_dailyHRV.csv       — daily HRV metrics
#   2_heartrate_1min.csv — per-minute heart rate readings
#
# To run:  python MainFitbit.py
# Both CSV files must be in the same folder as this script.
# =============================================================================

from FitbitFunctions import (
    # Task 1 — readers
    read_daily_hrv,
    read_heartrate_1min,
    # Task 2 — HRV summaries
    print_hrv_summary_table,
    # Task 2 — per-minute summaries
    print_heartrate_summary_table,
    compute_daily_stats,
    # Task 3 — visualizations
    plot_heartrate_one_day,
    plot_daily_hrv_metric,
    plot_cardiovascular_health,
)


# =============================================================================
# FILE PATHS — update these if your CSVs are in a different folder
# =============================================================================
HRV_FILE       = '2_dailyHRV.csv'
HEARTRATE_FILE = '2_heartrate_1min.csv'


# =============================================================================
# TASK 1 — Read both data files and report their shape
# =============================================================================
print("\n" + "="*55)
print("  TASK 1: Reading Data Files")
print("="*55)

hrv_df = read_daily_hrv(HRV_FILE)
print(f"\n  File: {HRV_FILE}")
print(f"    Rows    : {hrv_df.shape[0]}")
print(f"    Columns : {hrv_df.shape[1]}  {list(hrv_df.columns)}")

hr_df = read_heartrate_1min(HEARTRATE_FILE)
print(f"\n  File: {HEARTRATE_FILE}")
print(f"    Rows    : {hr_df.shape[0]:,}")
print(f"    Columns : {hr_df.shape[1]}  {list(hr_df.columns)}")
print(f"    Date range: {hr_df['Date'].min()}  →  {hr_df['Date'].max()}")


# =============================================================================
# TASK 2 — Process and summarize daily HRV metrics
# =============================================================================
print("\n" + "="*55)
print("  TASK 2: Processing & Summarizing")
print("="*55)

# --- 2_dailyHRV.csv ---
print("\n  [ 2_dailyHRV.csv ]")
print_hrv_summary_table(hrv_df)

# --- 2_heartrate_1min.csv ---
print("\n  [ 2_heartrate_1min.csv ]")
print_heartrate_summary_table(hr_df)


# =============================================================================
# TASK 3 — Visualize the data (three required plots)
# =============================================================================
print("\n" + "="*55)
print("  TASK 3: Visualizations")
print("="*55)

# --- Figure 1: Per-minute heart rate for a single given day ---
# Change DATE_TO_PLOT to any date in the dataset (YYYY-MM-DD format).
DATE_TO_PLOT = '2025-06-14'
print(f"\n  [Figure 1] Per-minute heart rate for {DATE_TO_PLOT}...")
plot_heartrate_one_day(hr_df, DATE_TO_PLOT)

# --- Figure 2: A daily HRV metric plotted over the recording period ---
# Change METRIC_TO_PLOT to any column from 2_dailyHRV.csv:
#   'daily_rmssd'  |  'deep_rmssd'  |  'resting_heart_rate'
METRIC_TO_PLOT = 'daily_rmssd'
print(f"\n  [Figure 2] Daily '{METRIC_TO_PLOT}' over the recording period...")
plot_daily_hrv_metric(hrv_df, metric=METRIC_TO_PLOT)

# --- Figure 3: Cardiovascular health zone distribution (pie chart) ---
# Each minute of heart rate data is classified into a zone based on BPM.
# The pie chart shows the proportion of time spent in each zone.
print("\n  [Figure 3] Cardiovascular health zone distribution...")
plot_cardiovascular_health(hr_df)

print("\nAll tasks complete.\n")