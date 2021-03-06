from scipy.stats import stats
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, f1_score, accuracy_score
from sklearn.linear_model import LinearRegression
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

# arousal
ar_df = df.filter(regex="^(AR)").astype("int32")
# valence
val_df = df.filter(regex="VA").astype("int32")

val1_df = df.filter(regex="VA01").astype("int32")
val2_df = df.filter(regex="VA02").astype("int32")
val3_df = df.filter(regex="VA03").astype("int32")
val4_df = df.filter(regex="VA04").astype("int32")
#print(val1_df.mean().values)
#print(fa_ly.transpose().mean()) need to differentiate between songs here!

ar_means = ar_df.mean()
val_means = val_df.mean()

plt.plot(val1_df.mean().values)
plt.plot(val2_df.mean().values)
plt.plot(val3_df.mean().values)
plt.plot(val4_df.mean().values)
plt.title("Valence mean for 4 songs")
plt.xlabel("Line")
plt.ylabel("Valence")
plt.legend(["val 1", "val 2", "val 3", "val 4"])
plt.show()



## Regressions
# BFI -> Valence Regression
ocean = data.read_ocean()
ocean['val'] = val_df.transpose().mean()#.values

#Y, X = patsy.dmatrices('val  ~ Openness + fa_so', ocean, return_type='dataframe')
#vif_df = pd.DataFrame()
#vif_df["vif"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
#vif_df["features"] = X.columns
#print(vif_df)
#x_fin = sm.add_constant(x) # adding a constant
#print(scipy.stats.spearmanr(x, y))
#ols_bfi = sm.OLS(endog=val_df.transpose().mean().values,
#                 exog=x_fin)  # have to add intercept term manually
#res_bfi = ols_bfi.fit()
#print(res_bfi.summary()) # all dimensions insignificant
#fig = plt.figure(figsize=(15,8))
#fig = sm.graphics.plot_partregress_grid(res_bfi, fig=fig)

#pd.pivot_table(ocean, values = 'val', index = 'Openness').plot.bar()
#plt.title("Openess and Valence")
#plt.show()


# Language -> Valence Regression
lang = data.read_language()
ols_lang = sm.OLS(endog=val_df.transpose().mean().values,
                  exog=sm.add_constant(lang["language"]), missing="drop")
res_lang = ols_lang.fit()
print(res_lang.summary())  # insignificant


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
vader_acc = accuracy_score(predictions_lines["val_sign"], predictions_lines["vader_label"]) #0.5825242718446602
sa_acc = accuracy_score(predictions_lines["val_sign"], predictions_lines["aap_label"]) #0.5825242718446602

sa_f1 = f1_score(predictions_lines["val_sign"], predictions_lines["aap_label"]) # 0.5057471264367815
vader_f1 = f1_score(predictions_lines["val_sign"], predictions_lines["vader_label"]) # 0.6906474820143884

# KNN
knn_val_sign = knn.KNeighborsClassifier()
knn_val_sign.fit(np.expand_dims(predictions_lines["aap"].array, axis=1), predictions_lines["val_sign"].array)
knn_val_sign.score(np.expand_dims(predictions_lines["aap"].array, axis=1), predictions_lines["val_sign"].array) # accuracy 0.7572815533980582

knn_val_sign.fit(np.expand_dims(predictions_lines["vader_compound"].array, axis=1), predictions_lines["val_sign"].array)
knn_val_sign.score(np.expand_dims(predictions_lines["vader_compound"].array, axis=1), predictions_lines["val_sign"].array) #accuracy 0.6019417475728155
