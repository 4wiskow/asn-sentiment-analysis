import statsmodels.api as sm
import data
import matplotlib.pyplot as plt
import seaborn as sns

val_open = data.all_val_openness()
val_open.corr()  # low r across all ppts
sns.lmplot(data=val_open, x="val", y="openness")
plt.show()

# Linear Regression: Openness, Group -> Valence
ols = sm.OLS(val_open["val"].array, sm.add_constant(val_open[["openness", "group"]].astype("float32")))
res = ols.fit()
res.summary()

