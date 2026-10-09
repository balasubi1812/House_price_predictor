import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("Bengaluru_House_Data.csv")

df = df[["location", "size", "total_sqft", "bath", "price"]].dropna()

df["bhk"] = df["size"].str.split().str[0].astype(int)

# Keep only plain numeric values
df["total_sqft"] = pd.to_numeric(df["total_sqft"], errors="coerce")
df = df.dropna()

df = df[df["total_sqft"] / df["bhk"] >= 300]

top = df["location"].str.strip().value_counts().head(20).index

df["location"] = df["location"].str.strip().where(
    df["location"].str.strip().isin(top),
    "other"
)

X = pd.get_dummies(
    df[["location", "total_sqft", "bath", "bhk"]],
    columns=["location"]
)

y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Average error: Rs", round(mean_absolute_error(y_test, pred), 1), "lakh")
print("R2 score:", round(r2_score(y_test, pred), 2))


def predict_price(location, sqft, bath, bhk):
    row = pd.DataFrame(0, index=[0], columns=X.columns)

    row["total_sqft"] = sqft
    row["bath"] = bath
    row["bhk"] = bhk

    col = "location_" + location

    if col in row.columns:
        row[col] = 1

    return model.predict(row)[0]


print("\nAvailable locations:", ", ".join(top))
print("(Type 'other' for any other area)\n")

while True:
    location = input("Location: ").strip()

    if location not in list(top) + ["other"]:
        print("Not in list, using 'other'")
        location = "other"

    sqft = float(input("Total sqft: "))
    bath = int(input("Bathrooms: "))
    bhk = int(input("Bedrooms (BHK): "))

    price = predict_price(location, sqft, bath, bhk)

    print("Estimated price: Rs", round(price, 1), "lakh\n")

    again = input("Predict another? (y/n): ").strip().lower()

    if again != "y":
        break