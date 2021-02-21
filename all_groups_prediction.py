import statsmodels.api as sm
import data
import matplotlib.pyplot as plt
import seaborn as sns

val_open = data.all_by_participant()
val_open.corr()  # some correlation for wellknown_dylan x valence
sns.lmplot(data=val_open, x="val", y="wellknown_dylan")
plt.show()

# Linear Regression: Openness, Group -> Valence
ols = sm.OLS(val_open["val"].array,
             sm.add_constant(val_open.loc[:, val_open.columns != "val"].astype("float32")))
res = ols.fit()
res.summary()  # wellknown dylan p = 0.02
