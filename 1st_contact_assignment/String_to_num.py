# 'Value', 'Wage' and "Release Clause' are string columns. Convert them to numbers. For eg, "M" in value column is Million, so multiply the row values by 1,000,000, etc
from cleaninig_dataset import df,pd
def converted(value):
    if pd.isna(value):
        return value
    
    value_str = str(value).strip()

    value_str.replace("€", " ").strip()

    if value_str.endswith("M"):
        try:
            num = float(value_str.replace("M", " ").strip()) * 1000000
            return num
        except:
            return value
    
    if value_str.endswith("k"):
        try:
            num = float(value_str.replace("k", " ").strip()) * 1000
            return num
        except:
            return value

if __name__ == "__main__":    
    stripped_columns = ["Value", "Wage", "Release Clause"]
    for col in stripped_columns:
        if col in df.columns:
            df[col].apply(converted)