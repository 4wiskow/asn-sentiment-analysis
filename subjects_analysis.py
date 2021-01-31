import csv
import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE_NAME = "data_Song_Lyrics_Gr6_2021-01-31_18-33.csv"

# read the data from the CSV
with open(CSV_FILE_NAME, "r", encoding="utf16") as csv_file:
    reader = csv.reader(csv_file, delimiter=",")
    header = next(reader)
    rows = [r for r in reader]
    df = pd.DataFrame(rows[1:], columns=header)

# BFI responses normally distributed?
bfi_df = df.filter(regex="BF02").astype("int32")
bfi_df.hist(xlabelsize=0, ylabelsize=10, sharey=True,)
plt.show()

# Dep Variable normally distributed across participants?
val_df = df.filter(regex="VA").astype("int32")
val_df.transpose().hist(xlabelsize=0, ylabelsize=10, sharey=True, figsize=(15, 10))
plt.show()

# Dep Variable normally distributed across participants?
ar_df = df.filter(regex="^(AR)").astype("int32")
ar_df.transpose().hist(xlabelsize=0, ylabelsize=10, sharey=True, figsize=(15, 10))
plt.show()

## Time
# sosci computed score for fast completion > 100 indicates low-quality data.
deg_time = df["DEG_TIME"].astype("int32")
deg_time[deg_time > 100].empty  # no values > 100, nothing to exclude

time_rsi = df["TIME_RSI"].astype("double")
time_rsi[time_rsi > 2.].empty  # no values > 2., nothing to exclude

# time taken for study should be more than 20 min
time_sum = df["TIME_SUM"].astype("int32")
time_sum[time_sum < 20*60].index  # subject 32 should be excluded

