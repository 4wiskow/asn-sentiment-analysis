from scipy.stats import stats
from seaborn import jointplot
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

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
sa_lines["val_z_ratings"] = stats.zscore(val_means.values)
sa_lines["ar_z_ratings"] = stats.zscore(ar_means.values)

# Metrics
r = sa_lines.corr().iloc[0, 1]  # do other metrics even make sense?
mse = mean_squared_error(sa_lines["aap"], sa_lines["val_z_ratings"])
r2 = r2_score(sa_lines["aap"], sa_lines["val_z_ratings"])

## Vader
vader_lines, vader_hit_rate = calc_vader_scores()
predictions_lines = pd.concat([sa_lines, vader_lines.drop("text", axis=1)], axis=1)
