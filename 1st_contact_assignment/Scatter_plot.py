import matplotlib.pyplot as plt
from cleaninig_dataset import df
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