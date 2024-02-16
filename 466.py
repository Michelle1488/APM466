import pandas as pd
from datetime import datetime, timedelta

data = pd.read_csv("C:\\Users\\Michelle\\Downloads\\466_DATA.csv")

price_data = data.iloc[:, 6:17]

# Calculate dirty price for a given bond
def find_dp(nth_bond, price_data):
    date1 = datetime(2024, 1, 8)
    date2 = datetime(2023, 9, 1)
    # Calculate the difference in days
    diff_days = (date1 - date2).days

    dp = []

    # Calculate dirty price for each day
    for i in range(len(price_data)):
        if i < 5:
            dp.append(((diff_days + i) / 365) * data.iloc[nth_bond-1, 1] + price_data.iloc[nth_bond-1, i])
            print(data.iloc[nth_bond-1, 1],price_data.iloc[nth_bond-1, i])
        elif 4 < i < 10:
            dp.append(((diff_days + 2 + i) / 365) * data.iloc[nth_bond-1, 1] + price_data.iloc[nth_bond-1, i])
            print(data.iloc[nth_bond - 1, 1], price_data.iloc[nth_bond - 1, i])
        elif i == 10:
            dp.append(((diff_days + 4 + i) / 365) * data.iloc[nth_bond-1, 1] + price_data.iloc[nth_bond-1, i])
            print(data.iloc[nth_bond - 1, 1], price_data.iloc[nth_bond - 1, i])

    return dp

# Calculate dirty price for each bond
dirty_price = []
for i in range(price_data.shape[1]):
    dp = find_dp(i+1, price_data)
    dirty_price.append(dp)

dirty_price_df = pd.DataFrame(dirty_price)

from sympy import symbols, Eq, exp, solve
import math

def find_eq(coupon, time, r):
    eq = 0
    for i in range(1, time + 1):
        eq += coupon/math.pow((1+r), i)
    eq += 100/math.pow((1+r), time)
    return eq

def find_ytm(price, coupon, time):
    coupon = coupon/100.0
    coupon_pmt = coupon*100

    ytm_r = coupon
    flag = True
    eq = find_eq(coupon_pmt / 2, time*2, ytm_r / 2)
    while flag:
        if (eq - price > 0.005):
            ytm_r += 0.00001
            # print(ytm_r, "eq>price")
        elif (eq - price < -0.005):
            ytm_r -= 0.00001
            # print(ytm_r, "eq<price")
        eq = find_eq(coupon_pmt/2, time*2, ytm_r/2)
        # print("the abs is :", abs(eq-price))
        if (abs(eq-price) <=0.005):
            flag = False
        else:
            flag = True
    return ytm_r

# print(data.iloc[:,1][0]) #this is the first coupon

rows = 10
cols = 10
ytm = []

for i in range(rows):
    row = []
    for j in range(cols):
        coupon = data.iloc[:, 1][j]
        price = dirty_price[j][i]
        time = 1+j
        row.append(find_ytm(price, coupon, time)*2*100)
    ytm.append(row)

print(pd.DataFrame(ytm))

import matplotlib.pyplot as plt
import numpy as np
x=["2024/3/1", "2024/9/1", "2025/3/1", "2025/9/1", "2026/3/1", "2026/9/1", "2027/3/1", "2027/9/1", "2028/3/1", "2028/9/1"]
plt.plot(x, ytm[0], label="Jan 8", marker='.')
plt.plot(x, ytm[1], label="Jan 9", marker='.')
plt.plot(x, ytm[2], label="Jan 10", marker='.')
plt.plot(x, ytm[3], label="Jan 11", marker='.')
plt.plot(x, ytm[4], label="Jan 12", marker='.')
plt.plot(x, ytm[5], label="Jan 15", marker='.')
plt.plot(x, ytm[6], label="Jan 16", marker='.')
plt.plot(x, ytm[7], label="Jan 17", marker='.')
plt.plot(x, ytm[8], label="Jan 18", marker='.')
plt.plot(x, ytm[9], label="Jan 19", marker='.')
plt.xlabel('Time')
plt.ylabel('Yield to Maturity (%)')
plt.title("5-Year Yield Curve")
plt.yticks(np.arange(2.5, 7.5, 0.5))
plt.legend()
plt.show()

# note that 10 >= day >= 1, 10 >= time >=1
def bootstrap(price, coupon, time, day):
    dates = [datetime(2024, 1, 8+day), datetime(2024, 3, 1), datetime(2024, 9, 1), datetime(2025, 3, 1), datetime(2025, 9, 1), datetime(2026, 3, 1),
             datetime(2026, 9, 1), datetime(2027, 3, 1), datetime(2027, 9, 1), datetime(2028, 3, 1), datetime(2028, 9, 1),
             datetime(2029, 3, 1)]
    if (time == 1):
        spot = math.log(price[0][day-1]/(0.5*coupon[0]+100))/-((dates[1]-dates[0]).days/365)
    elif (time == 2):
        spot = math.log(
            (price[1][day-1]-0.5 * coupon[1] * exp(-bootstrap(price, coupon, 1, day)*(dates[1]-dates[0]).days/365))
            /(0.5*coupon[1]+100))/-((dates[2]-dates[0]).days/365)
    elif (time == 3):
        spot = math.log((price[2][day-1]-0.5 * coupon[2] * exp(-bootstrap(price, coupon, 1, day)*(dates[1]-dates[0]).days/365)-0.5 * coupon[2] * exp(-bootstrap(price, coupon, 2, day)*(dates[2]-dates[0]).days/365))
            /(0.5*coupon[2]+100))/-((dates[3]-dates[0]).days/365)
    elif (time == 4):
        spot = math.log((price[3][day - 1] - 0.5 * coupon[3] * exp(
            -bootstrap(price, coupon, 1, day) * (dates[1] - dates[0]).days / 365) - 0.5 * coupon[3] * exp(
            -bootstrap(price, coupon, 2, day) * (dates[2] - dates[0]).days / 365) - 0.5 * coupon[3] * exp(
            -bootstrap(price, coupon, 3, day) * (dates[3] - dates[0]).days / 365))
                        / (0.5 * coupon[3] + 100)) / -((dates[4] - dates[0]).days / 365)
    elif (time == 5):
        spot = math.log((price[4][day - 1] - 0.5 * coupon[4] * exp(
            -bootstrap(price, coupon, 1, day) * (dates[1] - dates[0]).days / 365) - 0.5 * coupon[4] * exp(
            -bootstrap(price, coupon, 2, day) * (dates[2] - dates[0]).days / 365) - 0.5 * coupon[4] * exp(
            -bootstrap(price, coupon, 3, day) * (dates[3] - dates[0]).days / 365) - 0.5 * coupon[4] * exp(
            -bootstrap(price, coupon, 4, day) * (dates[4] - dates[0]).days / 365))
                        / (0.5 * coupon[4] + 100)) / -((dates[5] - dates[0]).days / 365)
    elif (time == 6):
        spot = math.log((price[5][day - 1] - 0.5 * coupon[5] * exp(
            -bootstrap(price, coupon, 1, day) * (dates[1] - dates[0]).days / 365) - 0.5 * coupon[5] * exp(
            -bootstrap(price, coupon, 2, day) * (dates[2] - dates[0]).days / 365) - 0.5 * coupon[5] * exp(
            -bootstrap(price, coupon, 3, day) * (dates[3] - dates[0]).days / 365) - 0.5 * coupon[5] * exp(
            -bootstrap(price, coupon, 4, day) * (dates[4] - dates[0]).days / 365) - 0.5 * coupon[5] * exp(
            -bootstrap(price, coupon, 5, day) * (dates[5] - dates[0]).days / 365))
                        / (0.5 * coupon[5] + 100)) / -((dates[6] - dates[0]).days / 365)
    elif (time == 7):
        spot = math.log((price[6][day - 1] - 0.5 * coupon[6] * exp(
            -bootstrap(price, coupon, 1, day) * (dates[1] - dates[0]).days / 365) - 0.5 * coupon[6] * exp(
            -bootstrap(price, coupon, 2, day) * (dates[2] - dates[0]).days / 365) - 0.5 * coupon[6] * exp(
            -bootstrap(price, coupon, 3, day) * (dates[3] - dates[0]).days / 365) - 0.5 * coupon[6] * exp(
            -bootstrap(price, coupon, 4, day) * (dates[4] - dates[0]).days / 365) - 0.5 * coupon[6] * exp(
            -bootstrap(price, coupon, 5, day) * (dates[5] - dates[0]).days / 365) - 0.5 * coupon[6] * exp(
            -bootstrap(price, coupon, 6, day) * (dates[6] - dates[0]).days / 365))
                        / (0.5 * coupon[6] + 100)) / -((dates[7] - dates[0]).days / 365)
    elif (time == 8):
        spot = math.log((price[7][day - 1] - 0.5 * coupon[7] * exp(
            -bootstrap(price, coupon, 1, day) * (dates[1] - dates[0]).days / 365) - 0.5 * coupon[7] * exp(
            -bootstrap(price, coupon, 2, day) * (dates[2] - dates[0]).days / 365) - 0.5 * coupon[7] * exp(
            -bootstrap(price, coupon, 3, day) * (dates[3] - dates[0]).days / 365) - 0.5 * coupon[7] * exp(
            -bootstrap(price, coupon, 4, day) * (dates[4] - dates[0]).days / 365) - 0.5 * coupon[7] * exp(
            -bootstrap(price, coupon, 5, day) * (dates[5] - dates[0]).days / 365) - 0.5 * coupon[7] * exp(
            -bootstrap(price, coupon, 6, day) * (dates[6] - dates[0]).days / 365) - 0.5 * coupon[7] * exp(
            -bootstrap(price, coupon, 7, day) * (dates[7] - dates[0]).days / 365))
                        / (0.5 * coupon[7] + 100)) / -((dates[8] - dates[0]).days / 365)
    elif (time == 9):
        spot = math.log((price[8][day - 1] - 0.5 * coupon[8] * exp(
            -bootstrap(price, coupon, 1, day) * (dates[1] - dates[0]).days / 365) - 0.5 * coupon[8] * exp(
            -bootstrap(price, coupon, 2, day) * (dates[2] - dates[0]).days / 365) - 0.5 * coupon[8] * exp(
            -bootstrap(price, coupon, 3, day) * (dates[3] - dates[0]).days / 365) - 0.5 * coupon[8] * exp(
            -bootstrap(price, coupon, 4, day) * (dates[4] - dates[0]).days / 365) - 0.5 * coupon[8] * exp(
            -bootstrap(price, coupon, 5, day) * (dates[5] - dates[0]).days / 365) - 0.5 * coupon[8] * exp(
            -bootstrap(price, coupon, 6, day) * (dates[6] - dates[0]).days / 365) - 0.5 * coupon[8] * exp(
            -bootstrap(price, coupon, 7, day) * (dates[7] - dates[0]).days / 365) - 0.5 * coupon[8] * exp(
            -bootstrap(price, coupon, 8, day) * (dates[8] - dates[0]).days / 365))
                        / (0.5 * coupon[8] + 100)) / -((dates[9] - dates[0]).days / 365)
    elif (time == 10):
        spot = math.log((price[9][day - 1] - 0.5 * coupon[9] * exp(
            -bootstrap(price, coupon, 1, day) * (dates[1] - dates[0]).days / 365) - 0.5 * coupon[9] * exp(
            -bootstrap(price, coupon, 2, day) * (dates[2] - dates[0]).days / 365) - 0.5 * coupon[9] * exp(
            -bootstrap(price, coupon, 3, day) * (dates[3] - dates[0]).days / 365) - 0.5 * coupon[9] * exp(
            -bootstrap(price, coupon, 4, day) * (dates[4] - dates[0]).days / 365) - 0.5 * coupon[9] * exp(
            -bootstrap(price, coupon, 5, day) * (dates[5] - dates[0]).days / 365) - 0.5 * coupon[9] * exp(
            -bootstrap(price, coupon, 6, day) * (dates[6] - dates[0]).days / 365) - 0.5 * coupon[9] * exp(
            -bootstrap(price, coupon, 7, day) * (dates[7] - dates[0]).days / 365) - 0.5 * coupon[9] * exp(
            -bootstrap(price, coupon, 8, day) * (dates[8] - dates[0]).days / 365) - 0.5 * coupon[9] * exp(
            -bootstrap(price, coupon, 9, day) * (dates[9] - dates[0]).days / 365))
                        / (0.5 * coupon[9] + 100)) / -((dates[10] - dates[0]).days / 365)
    elif (time == 11):
        spot = math.log((price[10][day - 1] - 0.5 * coupon[10] * exp(
            -bootstrap(price, coupon, 1, day) * (dates[1] - dates[0]).days / 365) - 0.5 * coupon[10] * exp(
            -bootstrap(price, coupon, 2, day) * (dates[2] - dates[0]).days / 365) - 0.5 * coupon[10] * exp(
            -bootstrap(price, coupon, 3, day) * (dates[3] - dates[0]).days / 365) - 0.5 * coupon[10] * exp(
            -bootstrap(price, coupon, 4, day) * (dates[4] - dates[0]).days / 365) - 0.5 * coupon[10] * exp(
            -bootstrap(price, coupon, 5, day) * (dates[5] - dates[0]).days / 365) - 0.5 * coupon[10] * exp(
            -bootstrap(price, coupon, 6, day) * (dates[6] - dates[0]).days / 365) - 0.5 * coupon[10] * exp(
            -bootstrap(price, coupon, 7, day) * (dates[7] - dates[0]).days / 365) - 0.5 * coupon[10] * exp(
            -bootstrap(price, coupon, 8, day) * (dates[8] - dates[0]).days / 365) - 0.5 * coupon[10] * exp(
            -bootstrap(price, coupon, 9, day) * (dates[9] - dates[0]).days / 365) - 0.5 * coupon[10] * exp(
            -bootstrap(price, coupon, 10, day) * (dates[10] - dates[0]).days / 365))
                        / (0.5 * coupon[10] + 100)) / -((dates[11] - dates[0]).days / 365)
    return spot

rows = 10
cols = 10
spr = []
percent = []
for i in range(rows):
    row = []
    temp = []
    for j in range(cols):
        coupon = data.iloc[:, 1]
        price = dirty_price
        time = 1+j
        day = 1+i
        row.append(bootstrap(price, coupon, time, day))
        temp.append(bootstrap(price, coupon, time, day) * 100)
    spr.append(row)
    percent.append(temp)

x=["2024/3/1", "2024/9/1", "2025/3/1", "2025/9/1", "2026/3/1", "2026/9/1", "2027/3/1", "2027/9/1", "2028/3/1", "2028/9/1"]
plt.plot(x, spr[0], label="Jan 8", marker='.')
plt.plot(x, spr[1], label="Jan 9", marker='.')
plt.plot(x, spr[2], label="Jan 10", marker='.')
plt.plot(x, spr[3], label="Jan 11", marker='.')
plt.plot(x, spr[4], label="Jan 12", marker='.')
plt.plot(x, spr[5], label="Jan 15", marker='.')
plt.plot(x, spr[6], label="Jan 16", marker='.')
plt.plot(x, spr[7], label="Jan 17", marker='.')
plt.plot(x, spr[8], label="Jan 18", marker='.')
plt.plot(x, spr[9], label="Jan 19", marker='.')
plt.xlabel('Time')
plt.ylabel('Spot Rate (%)')
plt.title("5-Year Spot Rate Curve")
plt.legend()
plt.show()

# time = 1,2, future = 3,4,...,10
def get_fwd(spot, time, future, day):
    diff = future - time
    eq = (spot[future-1][day-1]*future - spot[time-1][day-1]*time)/(diff/2)
    return eq

rows = 10
fwd = []
spot = spr
lst = [2,4,6,8]
for i in range(rows):
    row = []
    day = 1 + i
    for j in range(0,1):
        time = 1+j
        for num in lst:
            future = 1+num
            row.append(get_fwd(spot, time, future, day)*100)
    fwd.append(row)

x=["1yr-1yr", "1yr-2yr", "1yr-3yr", "1yr-4yr"]
plt.plot(x, fwd[0], label="Jan 8", marker='.')
plt.plot(x, fwd[1], label="Jan 9", marker='.')
plt.plot(x, fwd[2], label="Jan 10", marker='.')
plt.plot(x, fwd[3], label="Jan 11", marker='.')
plt.plot(x, fwd[4], label="Jan 12", marker='.')
plt.plot(x, fwd[5], label="Jan 15", marker='.')
plt.plot(x, fwd[6], label="Jan 16", marker='.')
plt.plot(x, fwd[7], label="Jan 17", marker='.')
plt.plot(x, fwd[8], label="Jan 18", marker='.')
plt.plot(x, fwd[9], label="Jan 19", marker='.')
plt.xlabel('Time (Periods)')
plt.ylabel('Forward Rate (%)')
plt.title("5-Year Forward Rate Curve")
plt.yticks(np.arange(6, 11, 0.5))
plt.legend()
plt.show()

print(pd.DataFrame(fwd))

matrix = []
for i in range(0, 5):
    temp = []
    for j in range(0, 9):
        x = math.log((ytm[j+1][i]) / (ytm[j][i]))
        temp.append(x)
    matrix.append(temp)
cov_mat = np.cov(matrix)
print(pd.DataFrame(cov_mat))

# Do the same for forward rates
matrix_fwd = []
for i in range(0, 4):
    temp = []
    for j in range(0, 9):
        x = math.log((fwd[j+1][i]) / (fwd[j][i]))
        temp.append(x)
    matrix_fwd.append(temp)
cov_mat_fwd = np.cov(matrix_fwd)
print(pd.DataFrame(cov_mat_fwd))

e_val1, e_vec1 = np.linalg.eig(cov_mat)
e_val2, e_vec2 = np.linalg.eig(cov_mat_fwd)
print(e_val1)
print(e_vec1)
print(e_val2)
print(e_vec2)