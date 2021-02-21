import csv
import pandas as pd
import re
import statsmodels.api as sm

CSV_FILE_NAME = "data/data_Song_Lyrics_Gr6_2021-01-31_18-33.csv"
GROUP_5_FNAME = "data/group5data.xlsx"
GROUP_7_FNAME = "data/group7data.xlsx"
#ALL_GROUPS = "data/test.xlsx"
DROP_PARTICIPANTS = [6, 17, 32, 39, 40, 42, 46]


def read_sosci(fname):
    """read the data from the CSV"""
    with open(fname, "r", encoding="utf16") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        header = next(reader)
        rows = [r for r in reader]
        df = pd.DataFrame(rows[1:], columns=header)

    df["group"] = "6"
    return df


def read_group_5():
    df = pd.read_excel(GROUP_5_FNAME,
                       engine='openpyxl')
    df = df.iloc[:38, ]  # select only participant data
    df["group"] = "5"
    return df


def read_group_7():
    df = pd.read_excel(GROUP_7_FNAME,
                       engine='openpyxl')

    df = df.iloc[:38, ]  # select only participant data
    df["group"] = "7"
    return df


def read_bfi_raw():
    """read raw bfi responses"""
    df = read_sosci(CSV_FILE_NAME)
    df = df.drop(DROP_PARTICIPANTS, axis=0)
    bfi_df = df.filter(regex="(BF02_)0?([1-9]|10)$").astype("int32")  # leave out mysterious 11th BFI question
    return bfi_df


def read_ocean():
    """read bfi and transform into OCEAN scores"""
    bfi_df = read_bfi_raw()
    ocean_pos = bfi_df.iloc[:, [9, 7, 5, 1, 8]]  # positively poled items, ordered by OCEAN

    ocean_neg = bfi_df.iloc[:, [4, 2, 0, 6, 3]]  # negatively poled items, ordered by OCEAN
    ocean_neg_repoled = (ocean_neg * -1) + 6  # invert negatively poled items for averaging

    ocean_pos.columns = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    ocean_neg_repoled.columns = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    ocean = (ocean_pos + ocean_neg_repoled) / 2  # average neg and pos poled items

    return ocean


def read_language():
    """read language (native english, native german, bilingual, other) and language level (B2 or lower, C1,
    C2) responses """
    df = read_sosci(CSV_FILE_NAME)
    df = df.drop(DROP_PARTICIPANTS, axis=0)
    lang_df = df.filter(regex="D01[2, 6]").astype("int32")
    lang_df.columns = ["language", "level"]
    lang_df = lang_df.replace(to_replace=-9, value=float("nan"))  # -9 indicates missing response

    return lang_df


def read_combined_data():
    df = read_sosci(CSV_FILE_NAME)
    df = df.drop(DROP_PARTICIPANTS, axis=0)
    df1 = read_group_5()
    df2 = read_group_7()

    # Rename all valence/Liking scores
    df2 = df2.rename(columns=lambda x: re.sub('^LL0', 'VA7_0', x))
    df1 = df1.rename(columns=lambda x: re.sub('^LN0', 'VA5_0', x))
    df = df.rename(columns=lambda x: re.sub('^VA0', 'VA6_0', x))

    # Rename all arousal/striking scores
    df2 = df2.rename(columns=lambda x: re.sub('^SL0', 'AR7_0', x))
    df1 = df1.rename(columns=lambda x: re.sub('^SN0', 'AR5_0', x))
    df = df.rename(columns=lambda x: re.sub('^AR0', 'AR6_0', x))

    # Rename BFI-scores Group 6
    df = df.rename(columns=lambda x: re.sub('^BF02', 'BF01', x))

    # Rename Song Knowledge in all groups
    df2 = df2.rename(columns=lambda x: re.sub('^SK0', 'SK7_0', x))
    df1 = df1.rename(columns=lambda x: re.sub('^SK0', 'SK5_0', x))
    df = df.rename(columns=lambda x: re.sub('^SK0', 'SK6_0', x))

    # rename group 7 native language
    df2 = df2.rename(columns={"D004": "D012"})

    # drop unimportant columns
    df2 = df2.loc[:, ~df2.columns.str.contains('^HI')]
    df1 = df1.loc[:, ~df1.columns.str.contains('^HI')]
    df1 = df1.loc[:, ~df1.columns.str.contains('^TIME')]
    df2 = df2.loc[:, ~df2.columns.str.contains('^TIME')]

    df

    new_df = pd.concat([df1, df2, df], ignore_index=True)

    # Rename descriptive colums in all groups
    new_df = new_df.rename(
        columns={
            "D001": "sex",
            "D002_01": "age",
            "D017": "highest_edu",
            "D012": "native_language",
            "DD01": "knowdylan",
            "DD02": "familiar_dylan",
            "DD03": "fan_dylan",
            "DD04": "wellknown_dylan",
            "DD05_01": "liking_gen_dylan",
        }
    )

    new_df["age"] = new_df["age"].astype("int32")
    new_df["wellknown_dylan"] = new_df["wellknown_dylan"].astype("int32")
    new_df["liking_gen_dylan"] = new_df["liking_gen_dylan"].astype("int32")

    sex = {1: "female", 2: "male"}
    new_df["native_language"] = new_df["native_language"].astype("int32")
    highest_edu = {2: "Highschool", 3: "Vocational degree", 4: "Bachelors", 5: "Masters", 6: "Doctorate"}
    native_language = {1: "English", 2: "German", 3: "English-German Bilingual", 4: "Other"}

    new_df['sex'].replace("1", "female", inplace=True)
    new_df['sex'].replace(1, "female", inplace=True)

    new_df['sex'] = new_df['sex'].replace("2", "male")
    new_df['sex'] = new_df['sex'].replace(2, "male")

    new_df["highest_edu_str"] = new_df["highest_edu"]
    new_df["native_language_str"] = new_df["native_language"]
    for key, value in highest_edu.items():
        new_df['highest_edu_str'].replace(str(key), value, inplace=True)
        new_df['highest_edu_str'].replace(key, value, inplace=True)

    for key, value in native_language.items():
        new_df['native_language_str'].replace(str(key), value, inplace=True)
        new_df['native_language_str'].replace(key, value, inplace=True)

    return new_df


def all_by_participant():
    """Get mean valence / liking and openness responses per participant by group"""
    cmb_df = read_combined_data()
    cmb_df = cmb_df[cmb_df["QUESTNNR"].isin(["lkng", "Liking", "qnr2"])]  # select ppts of 'liking' conditions
    #print(cmb_df["native_language"])
    #cmb_df["native_language"] = cmb_df["native_language"].astype("int32")
    fa5_1 = cmb_df["SK5_01"]
    fa5_2 = cmb_df["SK5_02"]
    fa5_3 = cmb_df["SK5_03"]
    fa5_4 = cmb_df["SK5_04"]
    fa6_1 = cmb_df["SK6_05_01"]
    fa6_2 = cmb_df["SK6_06_01"]
    fa6_3 = cmb_df["SK6_07_01"]
    fa6_4 = cmb_df["SK6_08_01"]
    fa7_1 = cmb_df["SK7_01_01"]
    fa7_2 = cmb_df["SK7_02_01"]
    fa7_3 = cmb_df["SK7_03_01"]
    liking = cmb_df \
        .filter(regex="VA") \
        .astype("float32") \
        .mean(axis=1)
    liking.name = "val"

    openness = cmb_df.filter(regex="BF01_0[9,4]").astype("int32")
    openness["BF01_04"] = (openness["BF01_04"] * -1) + 6
    openness = openness.mean(axis=1)
    openness.name = "openness"

    kd = cmb_df["knowdylan"]
    fd = cmb_df["familiar_dylan"]
    wd = cmb_df["wellknown_dylan"]
    lang = cmb_df["native_language"]
    val_open = pd.concat([liking, openness, cmb_df["group"], kd, fd, wd, fa5_1, fa5_2, fa5_3, fa5_4, fa6_1, fa6_2, fa6_3, fa6_4, fa7_1, fa7_2, fa7_3], axis=1)
    val_open = val_open.fillna(0)
    print(val_open.head())
    test = val_open.groupby(['group'])
    for name, group in test:
        if name == "7":
        # Linear Regression: Openness,familiarity -> Valence for first + 3rd song group 7 significant
            ols_1 = sm.OLS(group["val"].array, sm.add_constant(group.loc[:, ["openness","SK7_01_01"]].astype("float32")))
            res_1 = ols_1.fit()
            print(res_1.summary())
            fig = sm.graphics.plot_ccpr_grid(res_1)
            fig.suptitle('Openness-Familiarity for song 1')
            #fig = sm.graphics.plot_fit(res_1, "openness")

            ols_3 = sm.OLS(group["val"].array, sm.add_constant(group.loc[:, ["openness","SK7_03_01"]].astype("float32")))
            res_3 = ols_3.fit()
            print(res_3.summary())
            fig = sm.graphics.plot_ccpr_grid(res_3)
            fig.suptitle('Openness-Familiarity for song 3')

    val_open = val_open[["val", "openness", "group", "knowdylan", "familiar_dylan", "wellknown_dylan"]]
    return val_open
