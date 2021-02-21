import statsmodels.formula.api as sm
import data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np

val_open = data.all_by_participant()
plt.plot(val_open.index, val_open["val_z"])
plt.show()

val_open.corr()  # some correlation for wellknown_dylan x valence
sns.lmplot(data=val_open, y="val_z", x="wellknown_dylan", x_jitter=.1)
plt.show()

# Linear Regression: Openness, Group -> Valence
ols = sm.ols("val ~ wellknown_dylan + C(native_language_str) + C(group)", data=val_open)
res = ols.fit()
res.summary()  # wellknown dylan p < .5, native language p < .01


# NN
mlp = MLPRegressor((10,), activation='relu', random_state=1, max_iter=5000)
enc = OneHotEncoder()
d = val_open.loc[:, ["wellknown_dylan", "native_language_str", "group"]]
enc.fit(d)
x = enc.transform(d).toarray()
y = val_open.loc[:, "val"].to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

mlp.fit(x_train, y_train)
mlp.score(x_test, y_test)


## By Line
chords = data.read_chords()
plt.plot(chords["line"], chords["val_z_groups"])
plt.show()

chords[["AAPz", "val_z_groups"]].corr()
sns.jointplot(data=chords, x="chords", y="val_z_groups")
plt.show()

chords_ols = sm.ols("val_z_groups ~ AAPz + C(chords) + C(group_number)", data=chords).fit()
chords_ols.summary()  # AAPz and Chords Minor and Major/minor significant


mlp = MLPRegressor((100, 50), activation='relu', random_state=1, max_iter=1000)
enc = OneHotEncoder()
d = chords.loc[:, ["group_number", "chords", "group_number"]]
enc.fit(d)
x = enc.transform(d).toarray()
x = np.c_[x, chords.loc[:, "AAPz"]]  # add AAPz after dummy variables
y = chords.loc[:, "val"].to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

mlp.fit(x_train, y_train)
mlp.score(x_test, y_test)
