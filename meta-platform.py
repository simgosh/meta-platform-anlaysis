# libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

#importing original dataset 
#first review to dataset
meta = pd.read_csv("META.csv")
print(meta.head())
print(meta.info())
print(meta.isnull().sum())
print(meta.describe().T)

#Calculating differences for close price minus open price 
meta["Differences"] = meta["Close"] - meta["Open"]
print(meta[["Differences", "Close","Open"]])
print(meta.head())

#differences moving to next to close column 
first = meta.pop('Differences') 
meta.insert(5, 'Differences', first)
print(meta.columns) 


#convert to date from object to datetime 

meta["Date"] = pd.to_datetime(meta["Date"])
print(meta.info())

meta["Year"] = meta["Date"].dt.year
meta["Month"] = meta["Date"].dt.month
meta["Day"] = meta["Date"].dt.day

print(meta.head())
print(meta["Year"].value_counts())

#data visualization
fig = px.scatter(meta, x="Open", y="Close",
                hover_data="Date",
                color="High",
                color_continuous_scale=px.colors.sequential.Viridis
                 )
fig.show()

fig1 = px.scatter(meta,
                  x="Date",
                  y="Volume",
                  hover_data="Adj Close",
                  color="Close",
                  color_continuous_scale=px.colors.sequential.Sunsetdark)
fig1.show()

fig2 = px.scatter(meta,
                  x="High",
                  y="Close",
                  color="Month",
                  hover_data="Date",
                  color_continuous_scale=px.colors.sequential.Reds_r)
fig2.show()


avg_open_close_by_year= meta.groupby("Year")[["Close", "Open"]].mean().reset_index()
print(avg_open_close_by_year)


group_year = meta.groupby("Year")[["Open", "Close"]].agg(MAX_Open = ("Open", "max"),
                                                         MAX_Close = ("Close", "max"),
                                                         MIN_Open = ("Open", "min"),
                                                         MIN_Close=("Close", "min")
                                                        ).sort_values(by="Year", ascending=False).round(2)
print(group_year)

fig3 = px.bar(avg_open_close_by_year, x='Year',
              y=['Close', 'Open'], barmode="group",
              title="Open vs Close by Years",
              labels={"Year":"Year", "value":"Values"})
fig3.show()


fig4 = px.scatter(meta,
                  x="Open",
                  y="Close",
                  hover_data="Differences",
                  hover_name="Year",
                  size="Volume",
                  color="Volume",
                  color_continuous_scale=px.colors.sequential.RdPu_r,
                  title="Open vs Close Stock Price",
                  opacity=0.45,
                  template="plotly_dark")
fig4.show()


#Same graphs different values
fig5 = go.Figure()

fig5.add_trace(go.Scatter(x=meta["High"], y=meta["Low"], mode='lines', name='High-Low'))
fig5.add_trace(go.Scatter(x=meta["Open"], y=meta["Close"], mode="lines+markers", name = "Open-Close"))
fig5.show()


fig6 = px.scatter(meta,
                  x="Open",
                  y="Close",
                  hover_data="Differences",
                  facet_col="Year",
                  color="Volume",
                  size="Volume",
                  color_continuous_scale=px.colors.sequential.Peach_r,
                  template="seaborn")
fig6.show()


meta.drop("Date", axis=1, inplace=True)
print(meta.columns)

#Prediction with ML Models (Regression)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#split the data
X = meta.drop("Close", axis=1)
y = meta["Close"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

#Model training evulation 
models = {"Linear Regression":LinearRegression(),
          "RandomForestRegressor":RandomForestRegressor(random_state=42)
          }

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    #Evulate the model 
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'{name}: ')
    print(f'Mean Squared Error (MSE): {mse}')
    print(F'Root Mean Squared Error (RMSE) : {rmse}')
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'R-Squared Error: {r2}')
    print("--------------------------------------")
