import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

df = pd.read_csv("fifa21 raw data v2.csv")
# print(df.head())


df["Height"] = df["Height"].replace('[^0-9]', '', regex = True)
df["Height"] = df["Height"].astype(int)
# print(type(df["Height"].iloc[0]))

df["Weight"] = df["Weight"].replace("[^0-9]", '', regex = True)
df["Weight"] = df["Weight"].astype(int)
# print(type(df["Weight"].iloc[0]))

# print(df.head(13))

# remove unnecessary new line in all the columns 
df = df.map(lambda x: x.replace("\n", " ") if isinstance(x, str) else x)

# joined for more then ten years 
df["Joined"] = pd.to_datetime(df["Joined"], errors= "coerce")
Today = pd.Timestamp(datetime.today())

df["years_in_the_club"] = round((Today - df["Joined"]).dt.days / 365, 1)
long_term_players = df[df["years_in_the_club"] > 10]
# print(long_term_players[["Name", "Club", "Joined", "years_in_the_club"]].head())

# change M to number
def money_to_number(x):
    if isinstance(x, str):
        x = x.replace("€", '')
        if "M" in x:
            return float(x.replace("M", "")) * 1_000_000
        elif "K" in x:
            return float(x.replace("K", '')) * 1_000
        else:
            return float(x)
    return -1

for col in ["Value", "Wage", "Release Clause"]:
    df[col] = df[col].apply(money_to_number)

# print(df.head()["Value"])

# Some columns have 'star' characters. Strip those columns of these stars and make the columns numerical
star_col = [col for col in df.columns if df[col].astype(str).str.contains(r"★|\*").any()]

for col in star_col:
    df[col] = df[col].astype(str).str.replace("★", '', regex = False)
    df[col] = df[col].str.replace("*", "", regex = False)
    df[col] = pd.to_numeric(df[col], errors= "coerce")

# print(df.head()["W/F"])

# 6 Which players are highly valuable but still underpaid (on low wages)? (hint: scatter plot between wage and value)
plt.figure(figsize=(10,6))
plt.scatter(df["Wage"], df["Value"], alpha=0.3, color = "teal")

plt.title("Player Value Vs Wage")
plt.xlabel("Wage(€)")
plt.ylabel("value(€)")
plt.grid(True)
plt.show()

high_value = df["Value"] > df["Value"].quantile(0.75)
low_value = df["Wage"] > df["Wage"].quantile(0.25)

underpaid =  df[high_value & low_value]