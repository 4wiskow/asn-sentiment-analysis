from scipy.stats import stats
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, f1_score, accuracy_score
import pandas as pd
import seaborn as sns

from SentiArtBased import calc_aap
from vader import calc_vader_scores
from subjects_analysis import read_sosci, CSV_FILE_NAME, DROP_PARTICIPANTS

df = read_sosci(CSV_FILE_NAME)
df = df.drop(DROP_PARTICIPANTS, axis=0)
ar_df = df.filter(regex="^(AR)").astype("int32")
val_df = df.filter(regex="VA").astype("int32")

ar_means = ar_df.mean()
val_means = val_df.mean()

sa_lines = calc_aap()
sa_hit_rate = sa_lines["hit_rate"].mean()
sa_lines["aap_label"] = (sa_lines["aap"] >= 0).astype("int32")
sa_lines["aap_post_z"] = stats.zscore(sa_lines["aap"])
sa_lines["val_z_ratings"] = stats.zscore(val_means.values)
sa_lines["ar_z_ratings"] = stats.zscore(ar_means.values)


## Vader
vader_lines = calc_vader_scores()
vader_hit_rate = vader_lines["hit_rate"].mean()
predictions_lines = pd.concat([sa_lines, vader_lines.drop("text", axis=1)], axis=1)
#print("predictions_lines vader", predictions_lines)
#print("VADER", vader_hit_rate)

# Affective Time Series
aap_smooth = sa_lines.aap.rolling(window=5).mean()
val_smooth = sa_lines.val_z_ratings.rolling(window=5).mean()
plt.plot(range(len(sa_lines)), aap_smooth, val_smooth)
plt.title("AAP and Valence Time Series (Smoothed)")
plt.xlabel("Line")
plt.ylabel("y")
plt.legend(["aap", "val"])
plt.show()

## Metrics
# regression
r = predictions_lines[["val_z_ratings", "aap", "vader_compound"]].corr()
#predictions_lines[["val_z_ratings", "aap", "vader_compound"]].plot()
r2 = r2_score(sa_lines["aap_post_z"], sa_lines["val_z_ratings"])

# classification
predictions_lines["val_sign"] = (predictions_lines["val_z_ratings"] >= 0).astype("int32")
vader_acc = accuracy_score(predictions_lines["val_sign"], predictions_lines["vader_label"])
sa_acc = accuracy_score(predictions_lines["val_sign"], predictions_lines["aap_label"])

sa_f1 = f1_score(predictions_lines["val_sign"], predictions_lines["aap_label"])
vader_f1 = f1_score(predictions_lines["val_sign"], predictions_lines["vader_label"])