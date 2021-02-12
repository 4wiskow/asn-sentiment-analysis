from scipy.stats import stats
from seaborn import jointplot
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import seaborn as sns
from sklearn import linear_model
from SentiArtBased import calc_aap
from vader import calc_vader_scores
from subjects_analysis import read_sosci, CSV_FILE_NAME, DROP_PARTICIPANTS

df = read_sosci(CSV_FILE_NAME)
df = df.drop(DROP_PARTICIPANTS, axis=0)
ar_df = df.filter(regex="^(AR)").astype("int32")
val_df = df.filter(regex="VA").astype("int32")

ar_means = ar_df.mean()
val_means = val_df.mean()

sa_lines, sentiArt_hit_rate = calc_aap()
sa_lines["aap_post_z"] = stats.zscore(sa_lines["aap"])
sa_lines["val_z_ratings"] = stats.zscore(val_means.values)
sa_lines["ar_z_ratings"] = stats.zscore(ar_means.values)

# Metrics
r = sa_lines.corr().iloc[0, 1]  # do other metrics even make sense?
mse = mean_squared_error(sa_lines["aap_post_z"], sa_lines["val_z_ratings"])
r2 = r2_score(sa_lines["aap_post_z"], sa_lines["val_z_ratings"])
sns.lmplot(x="val_z_ratings", y="ar_z_ratings", data=sa_lines, order=2)
plt.tight_layout()
plt.show()

## Vader
vader_lines, vader_hit_rate = calc_vader_scores()
predictions_lines = pd.concat([sa_lines, vader_lines.drop("text", axis=1)], axis=1)

# Time Series
aap_smooth = sa_lines.aap_post_z.rolling(window=5).mean()
val_smooth = sa_lines.val_z_ratings.rolling(window=5).mean()
plt.plot(range(len(sa_lines)), aap_smooth, val_smooth)
plt.title("AAP and Valence Time Series (Smoothed)")
plt.xlabel("Line")
plt.ylabel("y")
plt.show()
