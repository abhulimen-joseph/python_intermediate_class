# Based on the 'Joined' column, check which players have been playing at a club for more than 10 years!
from datetime import datetime
from cleaninig_dataset import pd, df

# Convert 'Joined' to datetime
df['Joined'] = pd.to_datetime(df['Joined'], errors='coerce')

# Calculate years at club (as of 2021, since it's FIFA 21 data)
current_year = 2021
df['Years_At_Club'] = current_year - df['Joined'].dt.year

# Find players with > 10 years
long_serving_players = df[df['Years_At_Club'] > 10]

print(f"Players at club for >10 years: {len(long_serving_players)}")
print(long_serving_players[['Name', 'Club', 'Joined', 'Years_At_Club']].head())