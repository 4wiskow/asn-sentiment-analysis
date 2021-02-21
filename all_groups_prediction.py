import statsmodels.api as sm
import data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

val_open = data.all_by_participant()
val_open.corr()  # some correlation for wellknown_dylan x valence
sns.lmplot(data=val_open, x="val", y="native_language")
plt.show()

# Linear Regression: Openness, Group -> Valence
ols = sm.OLS(val_open["val"].array,
             sm.add_constant(val_open[["wellknown_dylan", "native_language", "group"]].astype("float32")))
res = ols.fit()
res.summary()  # wellknown dylan p < .5, native language p < .01

# NN
mlp = MLPRegressor(100, activation='relu', random_state=1, max_iter=500)
enc = OneHotEncoder()
d = val_open.loc[:, ["wellknown_dylan", "native_language", "group"]]
enc.fit(d)
x = enc.transform(d).toarray()
y = val_open.loc[:, "val"].to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

mlp.fit(x_train, y_train)
mlp.score(x_test, y_test)