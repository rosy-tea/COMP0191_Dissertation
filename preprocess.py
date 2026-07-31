import pandas as pd
import numpy as np
import re

df = pd.read_csv("perovskite_data.csv")

df_tmp = df.copy()

# ---------- ETL / HTL / BC ----------
def split_clean(x):
    if pd.isna(x): # nothing -> empty list
        return []
    
    parts = str(x).split("|") # split by "|"
    clean = []
    
    for p in parts: 
        p = p.strip() # if there are spaces around
        if p == "" or p.lower() in ["nan", "none", "unknown"]:
            continue
        clean.append(p) # "HTL1 | nan | HTL2" → ["HTL1", "HTL2"]
    
    return clean


def layer_match(df, layer_col, thk_col, prefix): # prefix: ETL / HTL / BC, ensuring each material layer has a thickness value

    df[f"{prefix}_layers"] = df[layer_col].apply(split_clean) # "ETL1 | ETL2" → ["ETL1", "ETL2"] -> material layers
    df[f"{prefix}_thickness_values"] = df[thk_col].apply(split_clean) # "100 | 200" → ["100", "200"] -> thickness values

    df[f"{prefix}_n_layers"] = df[f"{prefix}_layers"].apply(len)
    df[f"{prefix}_n_thickness"] = df[f"{prefix}_thickness_values"].apply(len)

    # define valid
    df[f"{prefix}_valid"] = (
        df[layer_col].notna() # 1) layer info and thickness info are both not empty
        & df[thk_col].notna()
        & (df[f"{prefix}_n_layers"] > 0)  # 2) number of layers > 0
        & (df[f"{prefix}_n_layers"] == df[f"{prefix}_n_thickness"])  # 3) number of layers == number of thickness values
    )
    
    return df

def parse_thickness_list_sum(values): # for baseline: simply sum up all thickness values in the list # only for ETL, HTL, BC
    nums = []

    for v in values:
        v = str(v).strip()
        if v == "" or v.lower() in ["nan", "none", "unknown"]: # if any value is unparseable, skip it
            continue

        try:
            num = float(v) # convert datatype to number
            if not np.isnan(num): # if it's a valid number, add to the list
                nums.append(num)
        except:
            continue

    if len(nums) == 0: # if all values are unparseable, return NaN
        return np.nan

    return sum(nums)

df_tmp = layer_match(df_tmp, "etl.stack_sequence", "etl.thickness", "ETL")
df_tmp = layer_match(df_tmp, "htl.stack_sequence", "htl.thickness_list", "HTL")
df_tmp = layer_match(df_tmp, "backcontact.stack_sequence", "backcontact.thickness_list", "BC")

df_tmp["ETL_thickness_clean"] = df_tmp["ETL_thickness_values"].apply(parse_thickness_list_sum)
df_tmp["HTL_thickness_clean"] = df_tmp["HTL_thickness_values"].apply(parse_thickness_list_sum)
df_tmp["BC_thickness_clean"] = df_tmp["BC_thickness_values"].apply(parse_thickness_list_sum)

df_tmp["ETL_valid"] = df_tmp["ETL_valid"] & df_tmp["ETL_thickness_clean"].notna()
df_tmp["HTL_valid"] = df_tmp["HTL_valid"] & df_tmp["HTL_thickness_clean"].notna()
df_tmp["BC_valid"] = df_tmp["BC_valid"] & df_tmp["BC_thickness_clean"].notna()

# ---------- PVK thickness ----------
# baseline: multiple values -> take the max value as the representative thickness
def parse_pvk_thickness(x):
    if pd.isna(x): # if empty, return NaN with note "missing"
        return pd.Series([np.nan, "missing_nan"])

    nums = []

    for p in str(x).split("|"): # multiple values separated by "|", we take the max value as the representative thickness, and note "multiple_values"
        p = p.strip()
        if p.lower() == "nan" or p == "":
            continue
        try:
            v = float(p)
            if not np.isnan(v):
                nums.append(v)
        except:
            continue

    if len(nums) == 0:
        return pd.Series([np.nan, "unparsed"])

    return pd.Series([max(nums), "max_value"])

df_tmp[["PVK_thickness_clean", "PVK_thickness_note"]] = (df_tmp["perovskite.thickness"].apply(parse_pvk_thickness))

df_tmp["PVK_valid"] = df_tmp["PVK_thickness_clean"].notna()

# all valid: only keep rows where all layers (ETL, HTL, PVK) are valid (no missing values)
df_tmp["all_layer_match"] = (df_tmp["ETL_valid"] & df_tmp["HTL_valid"] & df_tmp["PVK_valid"])


# range filtering: THK + bandgap (eliminate outliers and unreasonable values)
df_tmp = df_tmp[
    (df_tmp["ETL_thickness_clean"].between(5, 1000)) &
    (df_tmp["HTL_thickness_clean"].between(5, 1000)) &
    (df_tmp["PVK_thickness_clean"].between(50, 2000)) &
    (df_tmp["BC_thickness_clean"].between(10, 1000))
]

df_tmp["perovskite_bandgap_clean"] = pd.to_numeric(df_tmp["perovskite.band_gap"], errors="coerce")
df_tmp = df_tmp[df_tmp["perovskite_bandgap_clean"].between(1.0, 2.5)]


# JV parameter reasonable range filtering 
df_tmp = df_tmp[
    (df_tmp["jv.default_Jsc"].between(0, 45)) &
    (df_tmp["jv.default_Voc"].between(0, 1.3)) &
    (df_tmp["jv.default_FF"].between(0.3, 0.9))
]


# baseline complete: all valid + reasonable range for thickness, bandgap, JV parameters
df_tmp["baseline_complete"] = (
    df_tmp["PVK_thickness_clean"].notna() &
    df_tmp["ETL_thickness_clean"].notna() &
    df_tmp["HTL_thickness_clean"].notna() &
    df_tmp["perovskite_bandgap_clean"].notna() &
    df_tmp["jv.default_Jsc"].notna() &
    df_tmp["jv.default_Voc"].notna() &
    df_tmp["jv.default_FF"].notna() &
    df_tmp["jv.default_PCE"].notna()
)

df_base = df_tmp[df_tmp["baseline_complete"]].copy()

# Group by ref.ID and keep the entry with the highest PCE for each ref.ID (or DOI) to remove duplicates 
dedup_key = [
    "ref.ID",
    #"cell.stack_sequence",
    #"etl.stack_sequence",
    #"htl.stack_sequence",
    #"backcontact.stack_sequence",
    "PVK_thickness_clean",
    "ETL_thickness_clean",
    "HTL_thickness_clean",
    "perovskite_bandgap_clean",
    #"jv.default_Jsc",
    #"jv.default_Voc",
    #"jv.default_FF",
    "jv.default_PCE",
]

df_base = df_base.sort_values("jv.default_PCE", ascending=False)
df_base = df_base.drop_duplicates(subset=dedup_key)

print("baseline size:", len(df_base))

df_base.to_csv("df_base.csv", index=False) # for group A
df_base.to_pickle("df_base.pkl", index=False)