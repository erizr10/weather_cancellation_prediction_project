import pandas as pd
import numpy as np
import csv

#T_ONTIME_REPORTING.csv has shapes 538837, 10
#T_ONTIME_REPORTING-2.csv has shapes 59610, 10
#filtered_combined.csv has shapes 296774, 10

df = pd.read_csv('filtered_combined_3x.csv') #whole data

# filtered_data = df[df['FL_DATE'] == '1/7/2023 12:00:00 AM']
# cancelled_flights = np.array(filtered_data['CANCELLED'])
# total = np.sum(cancelled_flights)

#print(filtered_data)
#print("number of cancelled flights: " + str(total) + " out of " + str(cancelled_flights.size))

# print("File read. The shape and sizes are ")
# print(df.shape)

np_cancelled_flights = []
np_total_flights = []

list_of_dates = []
mydict = []

m = 1
d = 1
for i in range(1, 1096):
    d=i
    if (i > 31 and i <= 59):
        m = 2
        d -= 31
    elif (i > 59 and i <= 90):
        m = 3
        d -= 59
    elif (i > 90 and i <= 120):
        m = 4
        d -= 90
    elif (i > 120 and i <= 151):
        m = 5
        d -= 120
    elif (i > 151 and i <= 181):
        m = 6
        d -= 151
    elif (i > 181 and i <= 212):
        m = 7
        d -= 181
    elif (i > 212 and i <= 243):
        m = 8
        d -= 212
    elif (i > 243 and i <= 273):
        m = 9
        d -= 243
    elif (i > 273 and i <= 304):
        m = 10
        d -= 273
    elif (i > 304 and i <= 334):
        m = 11
        d -= 304
    elif (i > 334 and i <= 365):
        m = 12
        d -= 334
    
    current_date = str(m)+"/"+str(d)+"/2023 12:00:00 AM"
    list_of_dates.append(current_date)

    filtered_data = df[df['FL_DATE'] == current_date]
    filtered_data_can = filtered_data[filtered_data['CANCELLED'] == 1]
    filtered_data_can = filtered_data_can[filtered_data_can['CANCELLATION_CODE'].isin(['B', 'C'])]


    
    total = len(filtered_data.index)
    total_cancelled_flights = len(filtered_data_can.index)

    np_cancelled_flights.append(total_cancelled_flights)
    np_total_flights.append(total)


real_cancelled_flights = np.array(np_cancelled_flights)
real_total_flights = np.array(np_total_flights)

real_rate = np.array([real_cancelled_flights[i]/real_total_flights[i] for i in range(0, 365)])

for i in range(0, 365):
    temp = np.array([list_of_dates[i], real_rate[i]])
    mydict.append(temp)

print(real_rate)

filename = "date_and_cancellation_rate.csv"

with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    field = ['DATE', 'flight cancellation rate']
    
    writer.writerow(field)
    for i in range(0, 365):
        writer.writerow(mydict[i])


    #print(filtered_data)
    #print("number of cancelled flights: " + str(total) + " out of " + str(cancelled_flights.size))

#3/524, 14/621, 9/644, 10/612, 5/612, 0/614, 0/547, 