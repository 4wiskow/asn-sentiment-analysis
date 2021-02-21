from scipy.stats import stats
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, f1_score, accuracy_score
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import sklearn.neighbors as knn
from sklearn.linear_model import LogisticRegression
import numpy as np

from SentiArtBased import calc_aap
import data
from vader import calc_vader_scores

df = data.read_sosci(data.CSV_FILE_NAME)
df = df.drop(data.DROP_PARTICIPANTS, axis=0)
# familiarity with lyrics
fa_ly = df.filter(regex="LK").astype("int32")
# familiarity with song
fa_so = df.filter(regex="SK").astype("int32")
# arousal
ar_df = df.filter(regex="^(AR)").astype("int32")
# valence
val_df = df.filter(regex="VA").astype("int32")

ar_means = ar_df.mean()
val_means = val_df.mean()


## Regressions
# BFI -> Valence Regression
ocean = data.read_ocean()
ocean['val'] = val_df.transpose().mean().values
print(ocean.groupby(['Openness']).mean())
print(ocean.corr())
#x = ocean['Agreeableness']
#y = ocean['val']
#print(scipy.stats.spearmanr(x, y))
ols_bfi = sm.OLS(endog=val_df.transpose().mean().values,
                 exog=sm.add_constant(ocean))  # have to add intercept term manually
res_bfi = ols_bfi.fit()
print(res_bfi.summary()) # all dimensions insignificant
#fig = plt.figure(figsize=(15,8))
#fig = sm.graphics.plot_partregress_grid(res_bfi, fig=fig)

pd.pivot_table(ocean, values = 'val', index = 'Openness').plot.bar()
#plt.title("Openess and Valence")
plt.show()

# Language -> Valence Regression
lang = data.read_language()
ols_lang = sm.OLS(endog=val_df.transpose().mean().values,
                  exog=sm.add_constant(lang["language"]), missing="drop")
res_lang = ols_lang.fit()
res_lang.summary()  # insignificant


## SATs
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
# print("predictions_lines vader", predictions_lines)
# print("VADER", vader_hit_rate)

# Affective Time Series
aap_smooth = sa_lines.aap.rolling(window=5).mean()
val_smooth = sa_lines.val_z_ratings.rolling(window=5).mean()
plt.plot(range(len(sa_lines)), aap_smooth, val_smooth)
plt.title("AAP and Valence Time Series (Smoothed)")
plt.xlabel("Line")
plt.ylabel("y")
plt.legend(["aap", "val"])
plt.show()

# Metrics
# valence regression
r = predictions_lines[["val_z_ratings", "aap", "vader_compound"]].corr()
# predictions_lines[["val_z_ratings", "aap", "vader_compound"]].plot()
r2 = r2_score(sa_lines["aap_post_z"], sa_lines["val_z_ratings"])

# valence sign classification
predictions_lines["val_sign"] = (predictions_lines["val_z_ratings"] >= 0).astype("int32")
vader_acc = accuracy_score(predictions_lines["val_sign"], predictions_lines["vader_label"])
sa_acc = accuracy_score(predictions_lines["val_sign"], predictions_lines["aap_label"])

sa_f1 = f1_score(predictions_lines["val_sign"], predictions_lines["aap_label"])
vader_f1 = f1_score(predictions_lines["val_sign"], predictions_lines["vader_label"])

# KNN
knn_val_sign = knn.KNeighborsClassifier()
knn_val_sign.fit(np.expand_dims(predictions_lines["aap"].array, axis=1), predictions_lines["val_sign"].array)
knn_val_sign.score(np.expand_dims(predictions_lines["aap"].array, axis=1), predictions_lines["val_sign"].array)

knn_val_sign.fit(np.expand_dims(predictions_lines["vader_compound"].array, axis=1), predictions_lines["val_sign"].array)
knn_val_sign.score(np.expand_dims(predictions_lines["vader_compound"].array, axis=1), predictions_lines["val_sign"].array)

