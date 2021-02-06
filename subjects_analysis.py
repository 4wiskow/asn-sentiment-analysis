import csv
import pandas as pd
import matplotlib.pyplot as plt
from SentiArtBased import calc_aap
from vader import calc_vader_scores
from scipy import stats

CSV_FILE_NAME = "data/data_Song_Lyrics_Gr6_2021-01-31_18-33.csv"

# read the data from the CSV
with open(CSV_FILE_NAME, "r", encoding="utf16") as csv_file:
    reader = csv.reader(csv_file, delimiter=",")
    header = next(reader)
    rows = [r for r in reader]
    df = pd.DataFrame(rows[1:], columns=header)

## Time
# sosci-computed scores for fast completion
deg_time = df["DEG_TIME"].astype("int32")
deg_time[deg_time > 100].empty  # no values > 100, nothing to exclude

time_rsi = df["TIME_RSI"].astype("double")
time_rsi[time_rsi > 2.].empty  # no values > 2., nothing to exclude

# it should take at least 20 min to complete the study
time_sum = df["TIME_SUM"].astype("int32")
time_sum[time_sum < 20*60].index  # participant 32 should be excluded
time_sum.max() / 60  # longest time time to completion 60 min

## Response Data
# BFI responses normally distributed?
bfi_df = df.filter(regex="BF02").astype("int32")
bfi_df.hist(xlabelsize=0, ylabelsize=10, sharey=True,)
plt.title("BFI 10 Response Distributions")
plt.show()

# Dep Variable normally distributed across participants?
# Participants indicated for exclusion: 17, 39 and 40
ar_df = df.filter(regex="^(AR)").astype("int32")
ar_df.transpose().hist(xlabelsize=0, ylabelsize=10, sharey=True, figsize=(15, 10))
plt.suptitle("Arousal Response Distribution per Participant")
plt.show()

# Dep Variable normally distributed across participants?
# Participants indicated for exclusion: 17, 39 and 40
val_df = df.filter(regex="VA").astype("int32")
val_df.transpose().hist(xlabelsize=0, ylabelsize=10, sharey=True, figsize=(15, 10))
plt.suptitle("Valence Response Distribution per Participant")
plt.show()

# Arousal Mean across participants
ar_df.drop([17, 39, 40, 32], axis=0).mean().plot.density()
plt.title("Mean Arousal Response")
plt.show()
stats.shapiro(ar_df.mean())

# Valence Mean across participants
val_means = val_df.drop([17, 39, 40, 32], axis=0).mean()
val_means.plot.density()
plt.title("Mean Valence Response")
plt.show()
stats.shapiro(val_df.mean())  # p < .5, not normally distributed!

## Senti Art
sa_lines, sentiArt_hit_rate = calc_aap()
sa_lines["val_ratings"] = stats.zscore(val_means.values)
r = sa_lines.corr().iloc[0, 1]

## Vader
vader_lines, vader_hit_rate = calc_vader_scores()
predictions_lines = pd.concat([sa_lines, vader_lines.drop("text", axis=1)], axis=1)
