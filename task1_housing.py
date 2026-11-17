import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

homes = pd.read_csv("real_estate.csv")
homes["distance_100m"] = homes["X3 distance to the nearest MRT station"] / 100

X = homes[["distance_100m"]]
y = homes["Y house price of unit area"]

# (a) Fit
model = LinearRegression()
model.fit(X, y)
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# (b) Predict at 500 m
new_point = pd.DataFrame({"distance_100m": [5]})
print("Predicted price at 500 m:", model.predict(new_point)[0])

# (c) Plot
xs = pd.DataFrame({"distance_100m": [X["distance_100m"].min(), X["distance_100m"].max()]})
plt.scatter(X, y, alpha=0.5, label="Observations")
plt.plot(xs, model.predict(xs), color="red", label="Fitted line")
plt.xlabel("Distance to nearest MRT station (100 m)")
plt.ylabel("Price per unit area (10,000 NTD/ping)")
plt.title("Housing Price vs Distance to MRT")
plt.legend()
plt.savefig("task1_housing.png", dpi=150)
plt.show()