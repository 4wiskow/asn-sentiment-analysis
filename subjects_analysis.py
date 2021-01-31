import csv
import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE_NAME = "data_Song_Lyrics_Gr6_2021-01-31_18-33.csv"

# read the data from the CSV
with open(CSV_FILE_NAME, "r", encoding="utf16") as csv_file:
    reader = csv.reader(csv_file, delimiter=",")
    header = next(reader)
    rows = [r for r in reader]
    df = pd.DataFrame(rows[1:], columns=header)

# BFI responses normally distributed?
bfi_df = df.filter(regex="BF02").astype("int32")
bfi_df.hist(xlabelsize=0, ylabelsize=10, sharey=True,)
plt.show()

# Dep Variable normally distributed across participants?
val_df = df.filter(regex="VA").astype("int32")
val_df.transpose().hist(xlabelsize=0, ylabelsize=10, sharey=True, figsize=(15, 10))
plt.show()

# Dep Variable normally distributed across participants?
ar_df = df.filter(regex="^(AR)").astype("int32")
ar_df.transpose().hist(xlabelsize=0, ylabelsize=10, sharey=True, figsize=(15, 10))
plt.show()