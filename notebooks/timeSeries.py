#!/usr/bin/env python
# coding: utf-8

# ### F1 lap time prediction
# 
# Today, I have 24 hours to gather, analyse and train all race data (including lap times, circuit infos, pit_stops, etc) since the start of the 2011 season.
# The main objective is to make a prediction of the following lap time for a given car. 
# 
# If we succeed, we can try to expend the prediction to predict the lap time at lap l+n, l being the last lap completed (to predict the degradation, etc)
# 
# I already tried to resolve this issue using regression models, the result was really bad, let's try with some timeseries
# 
# Let start by gathering data from the "Formula 1 World Championship (1950 - 2023)" Kaggle : https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020

import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")


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
df_merged = df_merged[["raceId","year","name", "round", "lap", "driverRef", "driverId", "position", "time_x", "milliseconds", "date","time_y", "circuitId"]]
df_merged = df_merged[df_merged["time_y"]!=r"\N"]
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


# Let's try a really simple, one variable timeSerie resolution, we will use the lap as the "time" index

ts_OCO_ABU2022 = df_OCO_ABU2022[["date", "time_y","lap", "milliseconds"]]
#I volontarely add only one minute by lap
ts_OCO_ABU2022["ds"] = ts_OCO_ABU2022.apply(lambda row : np.datetime64(row.date+'T'+row.time_y)+np.timedelta64(row.lap,'m'), axis=1)
ts_OCO_ABU2022 = ts_OCO_ABU2022.rename(columns={"milliseconds":"y"}).reset_index(drop=True)[["ds",'y']]
ts_OCO_ABU2022.head()


ts_train = ts_OCO_ABU2022[:-4]
ts_test = ts_OCO_ABU2022[-4:]
ts_test_ds = ts_test[["ds"]]

print(ts_train.shape)
print(ts_test.shape)
print(ts_test_ds.shape)
ts_test


m = Prophet()
m.fit(ts_train)


forecast = m.predict(ts_test_ds)
forecast["y"] = ts_test["y"].reset_index(drop=True)
forecast["diff (s)"] = (forecast["yhat"]-forecast["y"])/1000

print("mean of the diff : ",np.mean(abs(forecast["diff (s)"])))

forecast[['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper', "diff (s)"]]


fig1 = m.plot(forecast)


# Half a second is way to high ! We know that in f1, this type of error can change everything... Let's try to drop the pitstops rows !

df_W_pit = pd.merge(df_merged, pit_stops, on=['raceId',"lap", "driverId"], how="left")

df_W_pit[np.isfinite(df_W_pit["stop"])]


toDrop = []
for name, group in df_W_pit.groupby(["raceId", "driverId"]):
    toDrop.append(group.index[0])
    for i in range(len(group)):
        if np.isfinite(group["stop"].iloc[i]):
            toDrop.extend([group.index[i], group.index[i+1] if i+1 < len(group) else None])

toDrop = [item for item in toDrop if item is not None]
print(len(toDrop))
df_W_pit.drop(index=toDrop,inplace=True)
df_W_pit.drop(columns=["time","duration","milliseconds_y","stop"], inplace=True)

df_W_pit


fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[1].plot(df_W_pit.index, df_W_pit['time_x'])
axs[1].set_xlabel('index')
axs[1].set_ylabel('lap time')

axs[0].plot(df_merged.index, df_merged['time_x'])
axs[0].set_xlabel('index')
axs[0].set_ylabel('lap_time')

plt.tight_layout()
plt.show()


# We can see that we don't have all the pit stops in our dataframe, but il should be enough

nb_laps_to_predict = 4
current_lap = 48

df_OCO_ABU2022 = df_W_pit.loc[(df_W_pit["year"]==2022) & (df_W_pit["name"]=="Abu Dhabi Grand Prix") & (df_W_pit["driverRef"]=="ocon")]

df_OCO_ABU2022 = df_OCO_ABU2022[["date", "time_y","lap", "milliseconds_x"]]
#I volontarely add only one minute by lap
df_OCO_ABU2022["ds"] = df_OCO_ABU2022.apply(lambda row : np.datetime64(row.date+'T'+row.time_y)+np.timedelta64(row.lap,'m'), axis=1)
df_OCO_ABU2022 = df_OCO_ABU2022.rename(columns={"milliseconds_x":"y"}).reset_index(drop=True)[["ds",'y']]

ts_train = df_OCO_ABU2022[:current_lap]
ts_test = df_OCO_ABU2022[current_lap:current_lap+nb_laps_to_predict]
ts_test_ds = ts_test[["ds"]]

m1 = Prophet()
m1.fit(ts_train)

forecast = m1.predict(ts_test_ds)
ts_test["yhat"] =  forecast["yhat"].values
ts_test["diff (s)"] = (ts_test["yhat"]-ts_test["y"])/1000

print("\nmean of the diff :",np.mean(abs(ts_test["diff (s)"])))
ts_test[['ds', 'y', 'yhat', "diff (s)"]]


fig2 = m1.plot(forecast)


fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(ts_test.index, ts_test['y'], label='True')
ax.plot(ts_test.index, ts_test['yhat'], label='Predicted')
ax.plot(ts_train.index, ts_train['y'], label='Previous')
ax.set_xlabel('Index')
ax.set_ylabel('Lap Time')
ax.legend()

plt.show()


# We can see that we obtain better results, but we should try to train with another model : arima

nb_laps_to_predict = 10
current_lap = 40

df_OCO_ABU2022 = df_W_pit.loc[(df_W_pit["year"]==2022) & (df_W_pit["name"]=="Abu Dhabi Grand Prix") & (df_W_pit["driverRef"]=="ocon")]

df_OCO_ABU2022 = df_OCO_ABU2022[["date", "time_y","lap", "milliseconds_x"]]
#I volontarely add only one minute by lap
df_OCO_ABU2022["ds"] = df_OCO_ABU2022.apply(lambda row : np.datetime64(row.date+'T'+row.time_y)+np.timedelta64(row.lap,'m'), axis=1)
df_OCO_ABU2022 = df_OCO_ABU2022.rename(columns={"milliseconds_x":"y"}).set_index("ds")[['y']]

ts_train = df_OCO_ABU2022[:current_lap]
ts_test = df_OCO_ABU2022[current_lap:current_lap+nb_laps_to_predict]

model = sm.tsa.ARIMA(ts_train['y'], order=(1,1,1))
model_fit = model.fit()

preds = model_fit.forecast(steps=nb_laps_to_predict)

ts_test["y_pred"] = preds[0]
ts_test["diff (s)"] = (ts_test["y_pred"]-ts_test["y"])/1000

print("mean of the diff :",np.mean(abs(ts_test["diff (s)"])))
ts_test[['y', 'y_pred', "diff (s)"]]


fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(ts_test.index, ts_test['y'], label='True')
ax.plot(ts_test.index, ts_test['y_pred'], label='Predicted')
ax.plot(ts_train.index, ts_train['y'], label='Previous')
ax.set_xlabel('Index')
ax.set_ylabel('Lap Time')
ax.legend()

plt.show()


# And, that's all, I gave myself one day to try that !
# 
# The prevision is not bad, of course we are limited by the data we have, with more data and more time, I'm sure we can find far better results !
