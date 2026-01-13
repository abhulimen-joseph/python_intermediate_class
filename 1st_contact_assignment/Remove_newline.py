# Remove the unnecessary newline characters from all columns that have them.
from cleaninig_dataset import df
df = df.replace("\n", " ", regex = True)
for col in df.columns:
    mask = df[col].astype(str).str.contains('\n', na= False)

    if mask.any():
        count = mask.sum()
        print(f"In {col} there are {count} new line characters ")

if mask.sum() == 0:
    print("There are no newline characters")