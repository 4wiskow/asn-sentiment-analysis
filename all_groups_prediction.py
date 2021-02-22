import statsmodels.formula.api as sm
import data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, KFold
import numpy as np

val_open = data.all_by_participant()
plt.plot(val_open.index, val_open["val_z"])
plt.show()

val_open.corr()  # some correlation for wellknown_dylan x valence
sns.lmplot(data=val_open, y="val_z", x="wellknown_dylan", x_jitter=.1)
plt.xticks([1., 2., 3., 4.], ["Not at all", "Somewhat", "Well", "Very Well"])
plt.title("Familiarity With Dylan vs. Valence")
plt.tight_layout()
plt.show()

sns.catplot(x="native_language", y="val_z", kind="box", data=val_open.replace("English-German Bilingual", "Bilingual"))
plt.title("Native Language vs. Valence Across All Groups")
plt.xlabel("Native Language")
plt.tight_layout()
plt.show()

# Linear Regression: Openness, Group -> Valence
val_open["val_z_abs"] = val_open["val_z"].abs()
ols = sm.ols("val_z ~ C(wellknown_dylan)  + C(native_language) + C(group)", data=val_open)
res = ols.fit()
res.summary()  # wellknown dylan p < .5, native language p < .01


# NN
enc = OneHotEncoder()
d = val_open.loc[:, ["wellknown_dylan", "native_language", "group"]]
enc.fit(d)
x = enc.transform(d).toarray()
y = val_open.loc[:, "val_z"].to_numpy()

kfold = KFold(5)
scores = []
for train_idx, test_idx in kfold.split(x, y):
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    mlp = MLPRegressor((100, 50), activation='relu', max_iter=5000)
    mlp.fit(x_train, y_train)
    s = mlp.score(x_test, y_test)
    scores.append(s)
np.mean(scores)


## By Line
chords = data.read_chords()
plt.plot(chords["line"], chords["val_z"])
plt.show()

chords[["AAPz", "val_z"]].corr()
sns.jointplot(data=chords, x="chords", y="val_z")
plt.show()

chords_ols = sm.ols("val_z ~ AAPz + C(chords) + C(group_number)", data=chords).fit()
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
