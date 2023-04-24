#!/usr/bin/env python
# coding: utf-8

# ### F1 lap time prediction
# 
# Today, I have 24 hours to gather, analyse and train all race data (including lap times, circuit infos, pit_stops, etc) since the start of the 2011 season.
# The main objective is to make a prediction of the following lap time for a given car. 
# 
# If we succeed, we can try to expend the prediction with two new features :
#     - Predict the lap time at lap l+n, l being the last lap completed (to predict the degradation, etc)
#     - Predict the lap time at lap l+n, with a pit stop at lap l+m with m<n
# 
# Let start by gathering data from the "Formula 1 World Championship (1950 - 2023)" Kaggle : https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

pd.options.mode.chained_assignment = None


lap_times = pd.read_csv("../data/lap_times.csv")
pit_stops = pd.read_csv("../data/pit_stops.csv")
races = pd.read_csv("../data/races.csv")
drivers = pd.read_csv("../data/drivers.csv")


# #### Lap times
# 
# Here is out main dataset !
# We can find every lap of every driver of every gp from the start of 1996 season to the end of 2022 season
# 
# Later, we should try to add 2023 first three GPs in the dataset

lap_times.sort_values("raceId")


# #### Races
# 
# This is the list of every gp from the start of 2009 to the end of 2023, it will be usefull to have some info about the circuit

races.sort_values(["year", "round"])


# #### Pit_stops
# 
# Same as the lap_times, we have here all the pit stops

pit_stops = pit_stops.sort_values("raceId").reset_index(drop=True)
pit_stops


# #### Drivers
# 
# Just a simple drivers list, with their ids

drivers = drivers.sort_values("driverId").reset_index(drop=True)
drivers


# Let's join everything we have !

df_merged = pd.merge(lap_times, drivers, on='driverId')
df_merged = pd.merge(df_merged, races, on='raceId')
#The race ids are not ordered by date
df_merged = df_merged.sort_values(["year", "round", "lap", "position"]).reset_index(drop=True)
df_merged = df_merged[["raceId","year","name", "round", "lap", "driverRef", "driverId", "position", "time_x", "milliseconds", "date", "circuitId"]]
df_merged["time_x"] = df_merged["time_x"].apply(lambda x: '0:' + x if len(x.split(':')) == 2 else x)
df_merged["time_x"] = pd.to_timedelta(df_merged['time_x'])
df_merged


# Let's plot it !

#First, we will watch the lap times of a specific driver, on a specific gp

df_OCO_ABU2022 = df_merged.loc[(df_merged["year"]==2022) & (df_merged["name"]=="Abu Dhabi Grand Prix") & (df_merged["driverRef"]=="ocon")]
df_OCO_ABU2022.head()


plt.plot(df_OCO_ABU2022["lap"], df_OCO_ABU2022["time_x"], label='Time by lap')
plt.ylabel('Time')
plt.xlabel('Lap Number')
plt.show()


# It's easy to see the two pit stops Ocon had. Let's drop those rows to have beter details

df_OCO_ABU2022_NOSTOP = df_OCO_ABU2022[df_OCO_ABU2022['time_x'] <= pd.Timedelta(minutes=1, seconds=35)]
plt.plot(df_OCO_ABU2022_NOSTOP["lap"], df_OCO_ABU2022_NOSTOP["milliseconds"], label='Time by lap')
plt.ylabel('Time')
plt.xlabel('Lap Number')
plt.show()


# As expected, we can see that pit stops are really important in lap time prediction, we are going to had a feature calculating the number of laps since last piststop

df_merged = pd.merge(df_merged, pit_stops, on=['raceId',"lap", "driverId"], how="left")

#actually we don't want to keep the races where there is no pit stops at all

listGpPit = pit_stops["raceId"].unique()
df_merged = df_merged[df_merged['raceId'].isin(listGpPit)]
df_merged.reset_index(drop=True, inplace=True)

df_merged = df_merged[["raceId","year","name", "round", "lap", "driverRef", "driverId", "position", "time_x", "milliseconds_x", "date", "circuitId", "stop"]]

df_merged


# Let's make a funcion computing the stops :

df = df_merged.copy()
df['stop'] = df.groupby(['raceId', 'driverRef'])['stop'].fillna(method='ffill')
df['stop'] = df['stop'].fillna(0).astype(int)
df


df["laps_since_last_stop"] = 0
df1 = pd.DataFrame()
for (raceId, driverId), df3 in df.groupby(["raceId", "driverId"]):
    cnt = 0
    nbStop = 0
    for i, row in df3.iterrows():
        cnt += 1
        if row["stop"] == nbStop:
            df3.at[i, "laps_since_last_stop"] = cnt
        else:
            nbStop += 1
            cnt = 0
    df1 = pd.concat([df1, df3], ignore_index=True)
df1


# Let's plot this for Ocon's Abu Dhabi 2022 GP

df_OCO_ABU2022 = df1[(df1["year"]==2022) & (df1["name"]=="Abu Dhabi Grand Prix") & (df1["driverRef"]=="ocon")]
df_OCO_ABU2022.head()


fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(df_OCO_ABU2022["lap"], df_OCO_ABU2022["stop"], label='Stops')
axes[0].set_ylabel('Number')
axes[0].set_xlabel('Lap')
axes[0].set_title('Stops')

axes[1].plot(df_OCO_ABU2022["lap"], df_OCO_ABU2022["laps_since_last_stop"], label='Laps since stop')
axes[1].set_ylabel('Number')
axes[1].set_xlabel('Lap')
axes[1].set_title('Laps since stop')

plt.tight_layout() 
plt.show()


# #### I think we are facing a timeSeries problem, but I will try to start with a simple regression to compare the results !
# 
# Let's one-hot encode some variables

#drop some useless columns
df_toEncode = df1.sort_values(["raceId","lap"]).reset_index(drop=True).drop(columns=["name","time_x", "driverRef", "date"])


"""print(df_toEncode.columns)

scaler = MinMaxScaler()
df_toEncode[['round', "lap", "position", "stop", "laps_since_last_stop"]] = scaler.fit_transform(df_toEncode[['round', "lap", "position", "stop", "laps_since_last_stop"]])

scaler = StandardScaler()
df_toEncode[['round', "lap", "position", "stop", "laps_since_last_stop"]] = scaler.fit_transform(df_toEncode[['round', "lap", "position", "stop", "laps_since_last_stop"]])

df_toEncode
"""


print(df_toEncode.columns)

# One-hot encode some columns
df_encoded = pd.get_dummies(df_toEncode, columns=["raceId", "year", "driverId", "circuitId"])

df_encoded.head()


# The correlations are very low, but let's keep that and train some models anyway !

#To test my program, I am going to take every last lap
race_cols = [col for col in df_encoded.columns if 'raceId' in col]
last_lap_indices = df_encoded.groupby(race_cols).apply(lambda x: x[x['lap'] == x['lap'].max()].index.tolist()).explode().tolist()
len(last_lap_indices)


test = df_encoded.iloc[last_lap_indices]
train = df_encoded.drop(last_lap_indices)
x_train, y_train = train.drop("milliseconds_x", axis=1),train["milliseconds_x"]
x_test, y_test = test.drop("milliseconds_x", axis=1),test["milliseconds_x"]
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

rf_regressor.fit(x_train, y_train)

y_pred = rf_regressor.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)


# We can see that the RMSE is way to high... as expected ! 

#Let's still look at some results
test["y_pred"] = y_pred
print("last GP estimation :")
test[test["raceId_1096"]==1][["milliseconds_x","y_pred"]]


# Catastrophic result, let's see in timeseries resolution

# 
