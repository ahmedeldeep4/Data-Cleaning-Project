import re
import numpy as np
import pandas as pd

INPUT = "raw_orders_dataset.csv"
OUTPUT = "clean_orders_dataset.csv"

# ---------- 1) Load ----------
df = pd.read_csv(INPUT, dtype=str, encoding="utf-8-sig")
rows_start = len(df)
print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ---------- 2) Clean column names ----------
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(r"[^\w]+", "_", regex=True)
    .str.strip("_")
)
# order_id, customer_name, age, city, email, phone_number, product_category,
# unit_price, quantity, order_date, payment_method, rating, returned, notes

# ---------- 3) Strip whitespace + unify missing values ----------
df = df.apply(lambda s: s.str.strip())
MISSING = ["", "n/a", "na", "null", "-", "?", "nan", "none"]
df = df.replace({v: np.nan for v in MISSING}, regex=False)
df = df.apply(lambda s: s.where(~s.str.lower().isin(MISSING)) if s.dtype == object else s)

# ---------- 4) Text standardization ----------
df["customer_name"] = df["customer_name"].str.title()

city_map = {
    "cairo": "Cairo", "cario": "Cairo", "القاهرة": "Cairo",
    "alex": "Alexandria", "alexandria": "Alexandria",
    "giza": "Giza", "gizah": "Giza",
    "mansoura": "Mansoura", "mansura": "Mansoura",
    "tanta": "Tanta", "aswan": "Aswan", "luxor": "Luxor",
}
df["city"] = df["city"].str.lower().map(city_map)

cat_map = {
    "electronics": "Electronics", "electronic": "Electronics",
    "clothing": "Clothing", "clothes": "Clothing",
    "home": "Home", "home & kitchen": "Home",
    "books": "Books", "book": "Books",
    "sports": "Sports", "sport": "Sports",
    "beauty": "Beauty",
}
df["product_category"] = df["product_category"].str.lower().map(cat_map)

pay_map = {
    "credit card": "Credit Card", "cc": "Credit Card",
    "cash on delivery": "Cash on Delivery", "cod": "Cash on Delivery", "cash": "Cash on Delivery",
    "fawry": "Fawry",
    "wallet": "Wallet", "mobile wallet": "Wallet",
}
df["payment_method"] = df["payment_method"].str.lower().map(pay_map)

bool_map = {"yes": True, "y": True, "true": True, "1": True,
            "no": False, "n": False, "false": False, "0": False}
df["returned"] = df["returned"].str.lower().map(bool_map)

df["notes"] = df["notes"].str.replace(r"\s+", " ", regex=True)

# ---------- 5) Numeric columns ----------
def to_number(s):
    return pd.to_numeric(s.str.replace(r"[^\d.\-]", "", regex=True), errors="coerce")

df["order_id"] = pd.to_numeric(df["order_id"], errors="coerce").astype("Int64")
df["age"] = to_number(df["age"])
df["unit_price"] = to_number(df["unit_price"])
df["quantity"] = to_number(df["quantity"])
df["rating"] = to_number(df["rating"])

# ---------- 6) Dates (mixed formats) ----------
def parse_date(x):
    if pd.isna(x):
        return pd.NaT
    x = x.strip()
    if re.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", x):      # 2024-01-15 / 2024/01/15
        fmt = "%Y-%m-%d" if "-" in x else "%Y/%m/%d"
    elif re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", x):          # 18/02/2025
        fmt = "%d/%m/%Y"
    elif re.fullmatch(r"\d{1,2}-\d{1,2}-\d{4}", x):          # 07-15-2025
        fmt = "%m-%d-%Y"
    elif re.fullmatch(r"\d{1,2} [A-Za-z]{3} \d{4}", x):      # 05 Mar 2024
        fmt = "%d %b %Y"
    else:
        return pd.NaT
    return pd.to_datetime(x, format=fmt, errors="coerce")

df["order_date"] = df["order_date"].apply(parse_date)

# ---------- 7) Validation / outliers -> NaN ----------
df.loc[~df["age"].between(18, 80), "age"] = np.nan
df.loc[~df["rating"].between(1, 5), "rating"] = np.nan
df.loc[~df["quantity"].between(1, 20), "quantity"] = np.nan
df.loc[df["unit_price"] <= 0, "unit_price"] = np.nan

q1, q3 = df["unit_price"].quantile([0.25, 0.75])
upper = q3 + 3 * (q3 - q1)                       # IQR rule (generous)
df.loc[df["unit_price"] > upper, "unit_price"] = np.nan

email_ok = df["email"].str.fullmatch(r"[\w.+-]+@[\w-]+\.[\w.]+", na=False)
df.loc[~email_ok, "email"] = np.nan
df["email"] = df["email"].str.lower()

def clean_phone(p):
    if pd.isna(p):
        return np.nan
    d = re.sub(r"\D", "", p)
    if d.startswith("20") and len(d) == 12:
        d = "0" + d[2:]
    return d if re.fullmatch(r"01[0125]\d{8}", d) else np.nan

df["phone_number"] = df["phone_number"].apply(clean_phone)

# ---------- 8) Remove duplicates ----------
before = len(df)
df = df.drop_duplicates(subset="order_id", keep="first")
print(f"Duplicates removed: {before - len(df)}")

# ---------- 9) Fill missing values ----------
df["age"] = df["age"].fillna(df["age"].median()).round().astype(int)
df["rating"] = df["rating"].fillna(df["rating"].median()).round().astype(int)
df["quantity"] = df["quantity"].fillna(df["quantity"].median()).round().astype(int)
df["unit_price"] = df["unit_price"].fillna(
    df.groupby("product_category")["unit_price"].transform("median")
).fillna(df["unit_price"].median()).round(2)

for col in ["city", "payment_method", "product_category"]:
    df[col] = df[col].fillna("Unknown")
df["returned"] = df["returned"].fillna(False)
df["notes"] = df["notes"].fillna("")

# ---------- 10) Derived column + save ----------
df["total_amount"] = (df["unit_price"] * df["quantity"]).round(2)
df = df.sort_values("order_id").reset_index(drop=True)
df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print(f"Rows: {rows_start} -> {len(df)}")
print("Remaining missing values:\n", df.isna().sum()[df.isna().sum() > 0])
print(f"Saved to {OUTPUT}")
