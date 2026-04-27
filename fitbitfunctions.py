# =============================================================================
# FitbitFunctions.py
# CSE-012 Course Project — Data Theme: Heart Rate
#
# All reusable functions for reading, processing, summarizing, and
# visualizing the two Fitbit heart rate datasets:
#
#   2_dailyHRV.csv       — daily HRV metrics (one row per day)
#   2_heartrate_1min.csv — per-minute heart rate readings
#
# MainFitbit.py calls these functions to produce all required outputs.
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# =============================================================================
# TASK 1 — READ DATA FILES
# =============================================================================

def read_daily_hrv(filepath):
    """
    Reads the daily HRV metrics CSV into a DataFrame.

    Columns in 2_dailyHRV.csv:
      date_pulled        — the calendar date of the recording
      daily_rmssd        — Root Mean Square of Successive Differences (ms),
                           computed across the full day. Higher RMSSD means
                           greater heart rate variability, which generally
                           indicates better recovery and cardiovascular health.
      deep_rmssd         — RMSSD measured specifically during deep sleep,
                           when the nervous system is most relaxed. Considered
                           the most reliable HRV window.
      resting_heart_rate — average BPM while at rest (not recorded every day).
                           A lower resting HR typically signals better
                           cardiovascular fitness.

    Parameters:
        filepath (str): path to 2_dailyHRV.csv

    Returns:
        pd.DataFrame: cleaned DataFrame with parsed dates
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    # Dates are stored as M/D/YY in the file (e.g. 5/21/25)
    df['date_pulled'] = pd.to_datetime(df['date_pulled'], format='%m/%d/%y')

    return df


def read_heartrate_1min(filepath):
    """
    Reads the per-minute heart rate CSV into a DataFrame.

    Columns in 2_heartrate_1min.csv:
      Time  — timestamp at minute resolution (date + hour + minute)
      Value — heart rate in beats per minute (BPM) for that minute

    Parameters:
        filepath (str): path to 2_heartrate_1min.csv

    Returns:
        pd.DataFrame: columns ['Time', 'Value', 'Date'] where Time is datetime
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    # The file has a double-space between date and time — strip before parsing
    df['Time'] = pd.to_datetime(df['Time'].str.strip(), format='%m/%d/%Y %I:%M:%S %p')

    # Add a date-only column for easy grouping and day-level filtering
    df['Date'] = df['Time'].dt.date

    return df


# =============================================================================
# TASK 2 — PROCESS & SUMMARIZE: 2_dailyHRV.csv
# =============================================================================

def count_days_hrv(hrv_df):
    """
    Returns how many unique days are recorded in 2_dailyHRV.csv.

    Parameters:
        hrv_df (pd.DataFrame): output of read_daily_hrv()

    Returns:
        int: number of days recorded
    """
    return hrv_df['date_pulled'].nunique()


def summarize_missing_hrv(hrv_df):
    """
    Counts how many days of data are missing for each HRV metric.
    A missing value is a blank cell (NaN) in the CSV.

    Parameters:
        hrv_df (pd.DataFrame): output of read_daily_hrv()

    Returns:
        pd.Series: number of missing days per metric column
    """
    metrics = ['daily_rmssd', 'deep_rmssd', 'resting_heart_rate']
    return hrv_df[metrics].isnull().sum()


def compute_hrv_stats(hrv_df):
    """
    Computes mean, maximum, and minimum for each daily HRV metric,
    skipping any missing values (NaN) in the calculation.

    Parameters:
        hrv_df (pd.DataFrame): output of read_daily_hrv()

    Returns:
        pd.DataFrame: rows = metrics, columns = [Mean, Maximum, Minimum]
    """
    metrics = ['daily_rmssd', 'deep_rmssd', 'resting_heart_rate']
    summary = pd.DataFrame({
        col: {
            'Mean':    round(hrv_df[col].mean(), 2),
            'Maximum': round(hrv_df[col].max(), 2),
            'Minimum': round(hrv_df[col].min(), 2),
        }
        for col in metrics
    }).T
    return summary


def print_hrv_summary_table(hrv_df):
    """
    Prints the full Task 2 summary for 2_dailyHRV.csv:
      - Number of days recorded
      - Missing data count per metric
      - Mean / Maximum / Minimum table for each metric

    Parameters:
        hrv_df (pd.DataFrame): output of read_daily_hrv()
    """
    print(f"\n  Days recorded in 2_dailyHRV.csv : {count_days_hrv(hrv_df)}")

    missing = summarize_missing_hrv(hrv_df)
    print("\n  Missing days per metric:")
    for col, count in missing.items():
        print(f"    {col:<22}: {int(count)} day(s) missing")

    stats = compute_hrv_stats(hrv_df)

    print("\n" + "="*62)
    print("  Daily HRV Summary Table (2_dailyHRV.csv)")
    print("="*62)
    print(f"{'Metric':<25} {'Mean':>10} {'Maximum':>10} {'Minimum':>10}")
    print("-"*62)
    for metric, row in stats.iterrows():
        print(f"{metric:<25} {row['Mean']:>10.2f} {row['Maximum']:>10.2f} {row['Minimum']:>10.2f}")
    print("="*62)


# =============================================================================
# TASK 2 — PROCESS & SUMMARIZE: 2_heartrate_1min.csv
# =============================================================================

def count_days_recorded(hr_df):
    """
    Returns how many unique dates are in the per-minute heart rate data.

    Parameters:
        hr_df (pd.DataFrame): output of read_heartrate_1min()

    Returns:
        int: number of unique days
    """
    return hr_df['Date'].nunique()


def summarize_missing_per_day(hr_df):
    """
    For each recorded date, calculates how many minute-readings are missing,
    assuming a full day should have 1,440 readings (60 min x 24 hrs).

    Parameters:
        hr_df (pd.DataFrame): output of read_heartrate_1min()

    Returns:
        pd.DataFrame: columns ['Date', 'Recorded', 'Missing']
    """
    counts = hr_df.groupby('Date')['Value'].count().reset_index()
    counts.columns = ['Date', 'Recorded']
    counts['Missing'] = 1440 - counts['Recorded']
    return counts


def compute_daily_stats(hr_df):
    """
    Computes mean, maximum, and minimum BPM for each recorded day
    from the per-minute heart rate data.

    Parameters:
        hr_df (pd.DataFrame): output of read_heartrate_1min()

    Returns:
        pd.DataFrame: columns ['Date', 'Mean_BPM', 'Max_BPM', 'Min_BPM']
    """
    stats = hr_df.groupby('Date')['Value'].agg(
        Mean_BPM='mean',
        Max_BPM='max',
        Min_BPM='min'
    ).reset_index()
    stats['Mean_BPM'] = stats['Mean_BPM'].round(1)
    return stats


def print_heartrate_summary_table(hr_df):
    """
    Prints the full Task 2 summary for 2_heartrate_1min.csv:
      - Number of days recorded
      - Missing minutes per day
      - Mean / Maximum / Minimum BPM per day, plus an overall row

    Parameters:
        hr_df (pd.DataFrame): output of read_heartrate_1min()
    """
    print(f"\n  Days recorded in 2_heartrate_1min.csv: {count_days_recorded(hr_df)}")

    missing_df = summarize_missing_per_day(hr_df)
    print("\n  Minutes recorded vs. missing per day:")
    print(f"  {'Date':<14} {'Recorded':>10} {'Missing':>10}")
    print("  " + "-"*36)
    for _, row in missing_df.iterrows():
        print(f"  {str(row['Date']):<14} {int(row['Recorded']):>10} {int(row['Missing']):>10}")

    stats_df = compute_daily_stats(hr_df)

    print("\n" + "="*55)
    print("  Heart Rate Summary Table (BPM per Day)")
    print("="*55)
    print(f"{'Date':<14} {'Mean BPM':>10} {'Max BPM':>10} {'Min BPM':>10}")
    print("-"*55)
    for _, row in stats_df.iterrows():
        print(f"{str(row['Date']):<14} {row['Mean_BPM']:>10.1f} "
              f"{int(row['Max_BPM']):>10} {int(row['Min_BPM']):>10}")
    print("-"*55)
    print(f"{'Overall':<14} {stats_df['Mean_BPM'].mean():>10.1f} "
          f"{int(stats_df['Max_BPM'].max()):>10} {int(stats_df['Min_BPM'].min()):>10}")
    print("="*55)


# =============================================================================
# TASK 3 — VISUALIZE
# =============================================================================

def slice_by_date(hr_df, date_str):
    """
    Filters the per-minute DataFrame to a single day.

    Parameters:
        hr_df (pd.DataFrame): output of read_heartrate_1min()
        date_str (str): date in 'YYYY-MM-DD' format, e.g. '2025-06-14'

    Returns:
        pd.DataFrame: rows for that date only, sorted by Time
    """
    target = pd.to_datetime(date_str).date()
    day_df = hr_df[hr_df['Date'] == target].copy()
    return day_df.sort_values('Time')


def plot_heartrate_one_day(hr_df, date_str):
    """
    Plots per-minute heart rate for a single given day (Figure 1).

    x-axis: time of day, formatted as HH:MM AM/PM for readability
    y-axis: heart rate in BPM

    Parameters:
        hr_df (pd.DataFrame): output of read_heartrate_1min()
        date_str (str): date in 'YYYY-MM-DD' format
    """
    day_df = slice_by_date(hr_df, date_str)

    if day_df.empty:
        print(f"  No data found for {date_str}. Check the date is in the dataset.")
        return

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(day_df['Time'], day_df['Value'],
            color='steelblue', linewidth=1.2, label='Heart Rate (BPM)')

    # Show a tick every 2 hours so the x-axis stays readable
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%I:%M %p'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    fig.autofmt_xdate(rotation=45)

    ax.set_title(f'Heart Rate per Minute — {date_str}', fontsize=13)
    ax.set_xlabel('Time of Day')
    ax.set_ylabel('Heart Rate (BPM)')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def plot_daily_hrv_metric(hrv_df, metric='daily_rmssd'):
    """
    Plots a chosen daily HRV metric across the full recording period (Figure 2).

    Valid metric options from 2_dailyHRV.csv:
      'daily_rmssd'        — overall daily heart rate variability
      'deep_rmssd'         — HRV during deep sleep
      'resting_heart_rate' — resting BPM (only on days it was recorded)

    x-axis: date
    y-axis: metric value

    Parameters:
        hrv_df (pd.DataFrame): output of read_daily_hrv()
        metric (str): column name to plot
    """
    valid = ['daily_rmssd', 'deep_rmssd', 'resting_heart_rate']
    if metric not in valid:
        print(f"  Invalid metric '{metric}'. Choose from: {valid}")
        return

    # Drop rows where this metric is NaN before plotting
    plot_df = hrv_df.dropna(subset=[metric]).copy()

    label_map = {
        'daily_rmssd':        'Daily RMSSD (ms)',
        'deep_rmssd':         'Deep Sleep RMSSD (ms)',
        'resting_heart_rate': 'Resting Heart Rate (BPM)'
    }

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.bar(plot_df['date_pulled'], plot_df[metric],
           color='steelblue', width=0.6,
           label=label_map[metric])

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    fig.autofmt_xdate(rotation=30)

    ax.set_title(f'{label_map[metric]} Over Recording Period', fontsize=13)
    ax.set_xlabel('Date')
    ax.set_ylabel(label_map[metric])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def classify_cardiovascular_health(hr_df):
    """
    Classifies every recorded minute into one of four cardiovascular zones
    based on standard BPM thresholds (American Heart Association guidelines):

        Resting    : BPM < 60    — very low, typically deep sleep
        Normal     : 60–99 BPM   — healthy resting / light activity
        Elevated   : 100–139 BPM — moderate exertion or mild stress
        High       : BPM >= 140  — vigorous activity or high load

    Parameters:
        hr_df (pd.DataFrame): output of read_heartrate_1min()

    Returns:
        pd.Series: zone label string for each row, same index as hr_df
    """
    def assign_zone(bpm):
        if bpm < 60:
            return 'Resting (<60)'
        elif bpm < 100:
            return 'Normal (60-99)'
        elif bpm < 140:
            return 'Elevated (100-139)'
        else:
            return 'High (>=140)'

    return hr_df['Value'].apply(assign_zone)


def plot_cardiovascular_health(hr_df):
    """
    Visualizes cardiovascular health zone distribution across the entire
    recording period using a pie chart.

    A pie chart is the most intuitive choice here: the goal is to show what
    proportion of all recorded minutes fell into each heart rate zone — a
    purely proportional question that pie charts communicate most clearly.

    Parameters:
        hr_df (pd.DataFrame): output of read_heartrate_1min()
    """
    zones = classify_cardiovascular_health(hr_df)
    zone_counts = zones.value_counts()

    ordered_labels = ['Resting (<60)', 'Normal (60-99)', 'Elevated (100-139)', 'High (>=140)']
    ordered_counts = [zone_counts.get(label, 0) for label in ordered_labels]
    colors = ['#5BA4CF', '#57A773', '#F4A259', '#E05C5C']

    fig, ax = plt.subplots(figsize=(7, 7))

    wedges, texts, autotexts = ax.pie(
        ordered_counts,
        labels=ordered_labels,
        autopct='%1.1f%%',
        colors=colors,
        startangle=140,
        pctdistance=0.78
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight('bold')

    ax.set_title('Cardiovascular Health Zone Distribution\n(entire recording period)', fontsize=13)
    plt.tight_layout()
    plt.show()