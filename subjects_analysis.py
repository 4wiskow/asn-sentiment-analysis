import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

from data import read_sosci

CSV_FILE_NAME = "data/data_Song_Lyrics_Gr6_2021-01-31_18-33.csv"

DROP_PARTICIPANTS = [6, 17, 32, 39, 40, 42, 46]

df = read_sosci(CSV_FILE_NAME)

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
#bfi_df.hist(xlabelsize=0, ylabelsize=10, sharey=True,)
#plt.title("BFI 10 Response Distributions")
#plt.tight_layout()
#plt.show()

ar_df = df.filter(regex="^(AR)").astype("int32")
ar_means = ar_df.drop(DROP_PARTICIPANTS, axis=0).mean()
val_df = df.filter(regex="VA").astype("int32")
val_means = val_df.drop(DROP_PARTICIPANTS, axis=0).mean()
ratings = pd.DataFrame(list(zip(val_means, ar_means)), columns=["val", "ar"])

# Dep Variable normally distributed across participants?
# Participants indicated for exclusion: 17, 39 and 40
#ar_df.transpose().hist(xlabelsize=0, ylabelsize=10, sharey=True, figsize=(15, 10))
#plt.suptitle("Arousal Response Distribution per Participant")
#plt.show()

ar_by_participant = ar_df.transpose().mean()
ar_by_participant[abs(stats.zscore(ar_by_participant)) > 2]  # no. 6 is beyond 2 sd, responded w/ 1 throughout song 2

# Dep Variable normally distributed across participants?
# Participants indicated for exclusion: 17, 39 and 40
#val_df.transpose().hist(xlabelsize=0, ylabelsize=10, sharey=True, figsize=(15, 10))
#plt.suptitle("Valence Response Distribution per Participant")
#plt.show()

# Participant 6 responded with 7 throughout song 2. Participants 42, 46 generally high values.
val_by_participant = val_df.transpose().mean()
val_by_participant[abs(stats.zscore(val_by_participant)) > 2]  # participants 6, 42, 46 are beyond 2 sd

# Arousal Mean across participants
#ar_means.plot.density()
#plt.title("Mean Arousal Response")
#plt.show()
stats.shapiro(ar_df.mean())


# Valence Mean across participants
#val_means.plot.density()
#plt.title("Mean Valence Response")
#plt.show()
stats.shapiro(val_means)  # p < .5, not normally distributed!

#h = jointplot(x=val_means.values, y=ar_means.values)
#h.set_axis_labels("valence ratings", "arousal ratings")
#plt.tight_layout()  # no inverse U?
#plt.show()

ratings = pd.DataFrame(list(zip(val_means, ar_means)), columns=["val", "ar"])
ratings["val_z"] = stats.zscore(ratings["val"])
ratings["ar_z"] = stats.zscore(ratings["ar"])
sns.lmplot(x="val_z", y="ar_z", data=ratings, order=2)
plt.tight_layout()
plt.show()