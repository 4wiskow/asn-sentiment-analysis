import csv
import pandas as pd

CSV_FILE_NAME = "data/data_Song_Lyrics_Gr6_2021-01-31_18-33.csv"
GROUP_5_FNAME = "data/group5data.xlsx"
GROUP_7_FNAME = "data/group7data.xlsx"
DROP_PARTICIPANTS = [6, 17, 32, 39, 40, 42, 46]


def read_sosci(fname):
    """read the data from the CSV"""
    with open(fname, "r", encoding="utf16") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        header = next(reader)
        rows = [r for r in reader]
        df = pd.DataFrame(rows[1:], columns=header)

    return df


def read_group_5():
    df = pd.read_excel(GROUP_5_FNAME)
    df = df.iloc[:38, ]  # select only participant data
    sr = df.filter(regex="SR0")
    sn = df.filter(regex="SN0")
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
