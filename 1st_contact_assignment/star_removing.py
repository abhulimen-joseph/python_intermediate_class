#4.  Some columns have 'star' characters. Strip those columns of these stars and make the columns numerical
from cleaninig_dataset import df,pd
def converted_num(value):
    if pd.isna(value):
        return value
    
    val_str = str(value).strip()
    if val_str.endswith("★"):
        try:
            num = float(val_str.replace("★"," ").strip())
            return num
        except:
            return value
if __name__ == "__main__":

    star_columns = []
    for col in df.columns:
        value = str(df[col].loc[0])
        if value.endswith("★"):
            star_columns.append(col)

        for col in star_columns:
            df[col] = df[col].apply(converted_num)
            print(df[col].loc[0])