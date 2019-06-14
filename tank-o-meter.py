# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 18:34:40 2019

@author: Asus
"""

import json
import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore
import math

with open("tankdata.json") as f:
    inputFile = json.load(f)

X2 = []
Y2 = []
x = []
y = []
X = []
H = []
count = 0
for i in inputFile['rows']:
    count = count + 1

print("Count:", count)
k = 0
i = 0

for i in inputFile['rows']:
    for j in i['doc']['data']:
        if j['tankName'] == 12:
            z = i['doc']['timestamp']
            X2.append(float(z))
            Y2.append(j['data'][2])
            k = k + 1

print('Max:', max(X2), 'time:', time.ctime(max(X2)))
print('Min:', min(X2), 'time:', time.ctime(min(X2)))
# plt.subplot(3,1,1)
# plt.plot(X2,Y2,'o',color='red') 
ax = plt.gca()
ax.set_ylim([10, 150])
ax.set_xlim([min(X2), max(X2)])

##########################################################
a = 0.01
theta = [68, 38]
for i in range(0, 5):
    sum1 = 0
    sum2 = 0

    for j in range(0, len(X2)):
        z = X2[j] * (max(X2) - min(X2) / 5) / (2 * math.pi)
        sum1 = sum1 + (theta[0] + theta[1] * math.sin(z) - Y2[j])
        sum2 = sum2 + (theta[0] + theta[1] * math.sin(z) - Y2[j]) * math.sin(z)

    t0 = theta[0] - a * (sum1 / len(X2))
    t1 = theta[1] - a * (sum2 / len(X2))
    theta[0] = t0
    theta[1] = t1
    print(theta)
print(len(X2))
print(theta)
print((max(X2) - min(X2) / 5) / (2 * math.pi))

tTheta = ((np.transpose(theta)))

for i in range(0, k):
    X.append(0)
    H.append(0)
    X[i] = X2[i]
    z = X[i] * (max(X2) - min(X2) / 5) / (2 * math.pi)
    H[i] = np.dot(tTheta, [1, math.sin(z)])
print(len(H))
print('Len:', len(X))
# plt.subplot(3,1,2)
# plt.plot(X,H,'o',color='green')
ax1 = plt.gca()
ax1.set_xlim([min(X2), max(X2)])
# ax1.set_ylim([30,50])

#############################################################

predict1 = []
T = []
i = 0
print(k)
k = max(X)
t = k
print('Max value:', k)
while True:
    k = k + 3600
    predict1.append(0)
    T.append(0)
    T[i] = k
    z = T[i] * (max(X2) - min(X2) / 5) / (2 * math.pi)
    predict1[i] = np.dot(tTheta, [1, math.sin(z)])
    if i == 120:
        break
    i = i + 1
# plt.subplot(3,1,3)
# plt.plot(T,predict)
ax2 = plt.gca()
ax2.set_xlim([t, k])

ax2.set_ylim([10, 150])
print(len(T), len(predict1))
# print(predict)
#############################################################
for j in range(0, 120):
    if predict1[j] < 40 and predict1[j] > 0:
        print(Fore.BLACK + ' Water level low at: ', time.ctime(T[j]), ',and the height is: ', predict1[j])
        print(Fore.RED + str(predict1[j]))
flag = 0

print(Fore.BLACK + ' ')
for j in range(0, 120):
    if (predict1[j] < 40 and predict1[j] > 0):
        td = time.time() - T[j]
        day = datetime.datetime.fromtimestamp(T[j]).strftime("%A")

        while True:
            if day == 'Saturday' or day == 'Sunday':
                if int(time.time()) >= int(T[j] + td + 900):
                    print('The water level is predicted to be low at ', time.ctime(T[j]),
                          ',but today is off,you need not turn on the motor. ')
                    flag = 2
                    break

            elif int(time.time()) >= int(T[j] + td + 1):
                print('Warning!! Water Level is predicted to be low at, ', time.ctime(T[j]), ' please fill your tank ')
                print('The height of tank is: ', predict1[j])
                recTime = time.time()
                print(' ')
                print('Please respond in 20 seconds! ')
                try:
                    for i in range(0, 20):
                        time.sleep(1)  # could use a backward counter to be preeety :)
                    print(' ')
                    print('Sorry! You have not responded.')
                    continue
                except KeyboardInterrupt:
                    input('Type anything:')
                    print('No more warnings.')
                    flag = 2
                    break
    if flag == 2:
        break
    if (predict1[j] >= 100):
        td = time.time() - T[j]
        p = 0
        while True:
            p = p + 1
            if p == 1:
                print('The water level is high enough ', time.ctime(T[j]))
                print('The height of tank is:', predict1[j])
                print(' ')
            if int(time.time()) == int(T[j] + td + 900):  # time.time()==T[j]-3600(when connected to the iOT device)
                print('The water level is high enough ', time.ctime(T[j]))
                print('The height of tank is: ', predict1[j])
                break
p1 = []
t1 = []
for i in range(24, 48):
    p1.append(predict1[i])
    z = (datetime.datetime.fromtimestamp(int(T[i])).strftime('%H'))
    t1.append(z)
colors = []
for i in p1:
    if i < 40:
        colors.append('r')
    else:
        colors.append('b')
plt.subplot(2, 1, 1)
plt.bar(t1, p1, color=colors)
plt.savefig('tank12day2.jpg')
plt.show()
######################################################## tank13
X2 = []
Y2 = []
x = []
y = []
X = []
H = []
count = 0
for i in inputFile['rows']:
    count = count + 1

print("Count:", count)
k = 0
i = 0

for i in inputFile['rows']:
    for j in i['doc']['data']:
        if j['tankName'] == 13:
            z = i['doc']['timestamp']
            X2.append(float(z))
            Y2.append(j['data'][2])
            k = k + 1

print('Max:', max(X2), 'time:', time.ctime(max(X2)))
print('Min:', min(X2), 'time:', time.ctime(min(X2)))
# plt.subplot(3,1,1)
# plt.plot(X2,Y2,'o',color='red') 
ax = plt.gca()
ax.set_ylim([10, 150])
ax.set_xlim([min(X2), max(X2)])

##########################################################
a = 0.01
theta = [68, 38]
for i in range(0, 5):
    sum1 = 0
    sum2 = 0

    for j in range(0, len(X2)):
        z = X2[j] * (max(X2) - min(X2) / 5) / (2 * math.pi)
        sum1 = sum1 + (theta[0] + theta[1] * math.sin(z) - Y2[j])
        sum2 = sum2 + (theta[0] + theta[1] * math.sin(z) - Y2[j]) * math.sin(z)

    t0 = theta[0] - a * (sum1 / len(X2))
    t1 = theta[1] - a * (sum2 / len(X2))
    theta[0] = t0
    theta[1] = t1
    print(theta)
print(len(X2))
print(theta)
print((max(X2) - min(X2) / 5) / (2 * math.pi))

tTheta = (np.transpose(theta))

for i in range(0, k):
    X.append(0)
    H.append(0)
    X[i] = X2[i]
    z = X[i] * (max(X2) - min(X2) / 5) / (2 * math.pi)
    H[i] = np.dot(tTheta, [1, math.sin(z)])
print(len(H))
print('Len:', len(X))
# plt.subplot(3,1,2)
# plt.plot(X,H,'o',color='green')
ax1 = plt.gca()
ax1.set_xlim([min(X2), max(X2)])
# ax1.set_ylim([30,50])

#############################################################

predict = []
T = []
i = 0
print(k)
k = max(X)
t = k
print('Max value:', k)
while True:
    k = k + 3600
    predict.append(0)
    T.append(0)
    T[i] = k
    z = T[i] * (max(X2) - min(X2) / 5) / (2 * math.pi)
    predict[i] = np.dot(tTheta, [1, math.sin(z)])
    if i == 120:
        break
    i = i + 1
# plt.subplot(3,1,3)
# plt.plot(T,predict)
ax2 = plt.gca()
ax2.set_xlim([t, k])

ax2.set_ylim([10, 150])
print(len(T), len(predict))
# print(predict)
#############################################################
for j in range(0, 120):
    if predict[j] < 40 and predict[j] > 0:
        print(Fore.BLACK + ' Water level low at:', time.ctime(T[j]), ',and the height is:', predict[j])
        print(Fore.RED + str(predict[j]))
flag = 0

print(Fore.BLACK + ' ')
for j in range(0, 120):
    if (predict[j] < 40 and predict[j] > 0):
        td = time.time() - T[j]
        day = datetime.datetime.fromtimestamp(T[j]).strftime("%A")

        while True:
            if day == 'Saturday' or day == 'Sunday':
                if int(time.time()) >= int(T[j] + td + 900):
                    print('The water level is predicted to be low at ', time.ctime(T[j]),
                          ',but today is off,you need not turn on the motor.')
                    flag = 2
                    break

            elif int(time.time()) >= int(T[j] + td + 1):
                print('Warning!!Water Level is predicted to be low at,', time.ctime(T[j]), ' please fill your tank ')
                print('The height of tank is: ', predict[j])
                recTime = time.time()
                print(' ')
                print('Please respond in 20 seconds! ')
                try:
                    for i in range(0, 20):
                        time.sleep(1)  # could use a backward counter to be preeety :)
                    print(' ')
                    print('Sorry! You have not responded.')
                    continue
                except KeyboardInterrupt:
                    input('Type anything:')
                    print('No more warnings!')
                    flag = 2
                    break
    if flag == 2:
        break
    if (predict[j] >= 100):
        td = time.time() - T[j]
        p = 0
        while True:
            p = p + 1
            if p == 1:
                print('The water level is high enough ', time.ctime(T[j]))
                print('The height of tank is:', predict[j])
                print(' ')
            if int(time.time()) == int(T[j] + td + 900):  # time.time()==T[j]-3600(when connected to the iOT device)
                print('The water level is high enough ', time.ctime(T[j]))
                print('The height of tank is:', predict[j])
                break
p2 = []
t2 = []

for i in range(24, 48):
    p2.append(predict[i])
    z = (datetime.datetime.fromtimestamp(int(T[i])).strftime('%H'))
    t2.append(z)
colors = []
for i in p2:
    if i < 40:
        colors.append('r')
    else:
        colors.append('b')
plt.subplot(2, 1, 2)
plt.bar(t2, p2, color=colors)
plt.savefig('tank13day2.jpg')
plt.show()
import json
import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore
import math

with open("tankdata.json") as f:
    inputFile = json.load(f)

X2 = []
Y2 = []
x = []
y = []
X = []
H = []
count = 0
for i in inputFile['rows']:
    count = count + 1

print("Count:", count)
k = 0
i = 0

for i in inputFile['rows']:
    for j in i['doc']['data']:
        if j['tankName'] == 12:
            z = i['doc']['timestamp']
            X2.append(float(z))
            Y2.append(j['data'][2])
            k = k + 1

print('Max:', max(X2), 'time:', time.ctime(max(X2)))
print('Min:', min(X2), 'time:', time.ctime(min(X2)))
# plt.subplot(3,1,1)
# plt.plot(X2,Y2,'o',color='red') 
ax = plt.gca()
ax.set_ylim([10, 150])
ax.set_xlim([min(X2), max(X2)])

##########################################################
a = 0.01
theta = [68, 38]
for i in range(0, 5):
    sum1 = 0
    sum2 = 0

    for j in range(0, len(X2)):
        z = X2[j] * (max(X2) - min(X2) / 5) / (2 * math.pi)
        sum1 = sum1 + (theta[0] + theta[1] * math.sin(z) - Y2[j])
        sum2 = sum2 + (theta[0] + theta[1] * math.sin(z) - Y2[j]) * math.sin(z)

    t0 = theta[0] - a * (sum1 / len(X2))
    t1 = theta[1] - a * (sum2 / len(X2))
    theta[0] = t0
    theta[1] = t1
    print(theta)
print(len(X2))
print(theta)
print((max(X2) - min(X2) / 5) / (2 * math.pi))

tTheta = ((np.transpose(theta)))

for i in range(0, k):
    X.append(0)
    H.append(0)
    X[i] = X2[i]
    z = X[i] * (max(X2) - min(X2) / 5) / (2 * math.pi)
    H[i] = np.dot(tTheta, [1, math.sin(z)])
print(len(H))
print('Len:', len(X))
# plt.subplot(3,1,2)
# plt.plot(X,H,'o',color='green')
ax1 = plt.gca()
ax1.set_xlim([min(X2), max(X2)])
# ax1.set_ylim([30,50])

#############################################################

predict1 = []
T = []
i = 0
print(k)
k = max(X)
t = k
print('Max value:', k)
while True:
    k = k + 3600
    predict1.append(0)
    T.append(0)
    T[i] = k
    z = T[i] * (max(X2) - min(X2) / 5) / (2 * math.pi)
    predict1[i] = np.dot(tTheta, [1, math.sin(z)])
    if i == 120:
        break
    i = i + 1
# plt.subplot(3,1,3)
# plt.plot(T,predict)
ax2 = plt.gca()
ax2.set_xlim([t, k])

ax2.set_ylim([10, 150])
print(len(T), len(predict1))
# print(predict)
#############################################################
for j in range(0, 120):
    if predict1[j] < 40 and predict1[j] > 0:
        print(Fore.BLACK + ' Water level low at:', time.ctime(T[j]), ',and the height is:', predict1[j])
        print(Fore.RED + str(predict1[j]))
flag = 0

print(Fore.BLACK + ' ')
for j in range(0, 120):
    if (predict1[j] < 40 and predict1[j] > 0):
        td = time.time() - T[j]
        day = datetime.datetime.fromtimestamp(T[j]).strftime("%A")

        while True:
            if day == 'Saturday' or day == 'Sunday':
                if int(time.time()) >= int(T[j] + td + 900):
                    print('The water level is predicted to be low at', time.ctime(T[j]),
                          ',but today is off,you need not turn on the motor.')
                    flag = 2
                    break

            elif int(time.time()) >= int(T[j] + td + 1):
                print('Warning!!Water Level is predicted to be low at,', time.ctime(T[j]), ' please fill your tank ')
                print('The height of tank is:', predict1[j])
                recTime = time.time()
                print(' ')
                print('Please respond in 20 seconds! ')
                try:
                    for i in range(0, 20):
                        time.sleep(1)  # could use a backward counter to be preeety :)
                    print(' ')
                    print('Sorry!You have not responded.')
                    continue
                except KeyboardInterrupt:
                    input('Type anything:')
                    print('No more warnings!')
                    flag = 2
                    break
    if flag == 2:
        break
    if (predict1[j] >= 100):
        td = time.time() - T[j]
        p = 0
        while True:
            p = p + 1
            if p == 1:
                print('The water level is high enough ', time.ctime(T[j]))
                print('The height of tank is:', predict1[j])
                print(' ')
            if int(time.time()) == int(T[j] + td + 900):  # time.time()==T[j]-3600(when connected to the iOT device)
                print('The water level is high enough ', time.ctime(T[j]))
                print('The height of tank is:', predict1[j])
                break
p1 = []
t1 = []
for i in range(24, 48):
    p1.append(predict1[i])
    z = (datetime.datetime.fromtimestamp(int(T[i])).strftime('%H'))
    t1.append(z)
colors = []
for i in p1:
    if i < 40:
        colors.append('r')
    else:
        colors.append('b')
plt.subplot(2, 1, 1)
plt.bar(t1, p1, color=colors)
plt.savefig('tank12day2.jpg')
plt.show()
######################################################## tank13
X2 = []
Y2 = []
x = []
y = []
X = []
H = []
count = 0
for i in inputFile['rows']:
    count = count + 1

print("Count:", count)
k = 0
i = 0

for i in inputFile['rows']:
    for j in i['doc']['data']:
        if j['tankName'] == 13:
            z = i['doc']['timestamp']
            X2.append(float(z))
            Y2.append(j['data'][2])
            k = k + 1

print('Max:', max(X2), 'time:', time.ctime(max(X2)))
print('Min:', min(X2), 'time:', time.ctime(min(X2)))
# plt.subplot(3,1,1)
# plt.plot(X2,Y2,'o',color='red') 
ax = plt.gca()
ax.set_ylim([10, 150])
ax.set_xlim([min(X2), max(X2)])

##########################################################
a = 0.01
theta = [68, 38]
for i in range(0, 5):
    sum1 = 0
    sum2 = 0

    for j in range(0, len(X2)):
        z = X2[j] * (max(X2) - min(X2) / 5) / (2 * math.pi)
        sum1 = sum1 + (theta[0] + theta[1] * math.sin(z) - Y2[j])
        sum2 = sum2 + (theta[0] + theta[1] * math.sin(z) - Y2[j]) * math.sin(z)

    t0 = theta[0] - a * (sum1 / len(X2))
    t1 = theta[1] - a * (sum2 / len(X2))
    theta[0] = t0
    theta[1] = t1
    print(theta)
print(len(X2))
print(theta)
print((max(X2) - min(X2) / 5) / (2 * math.pi))

tTheta = (np.transpose(theta))

for i in range(0, k):
    X.append(0)
    H.append(0)
    X[i] = X2[i]
    z = X[i] * (max(X2) - min(X2) / 5) / (2 * math.pi)
    H[i] = np.dot(tTheta, [1, math.sin(z)])
print(len(H))
print('Len:', len(X))
# plt.subplot(3,1,2)
# plt.plot(X,H,'o',color='green')
ax1 = plt.gca()
ax1.set_xlim([min(X2), max(X2)])
# ax1.set_ylim([30,50])

#############################################################

predict = []
T = []
i = 0
print(k)
k = max(X)
t = k
print('Max value:', k)
while True:
    k = k + 3600
    predict.append(0)
    T.append(0)
    T[i] = k
    z = T[i] * (max(X2) - min(X2) / 5) / (2 * math.pi)
    predict[i] = np.dot(tTheta, [1, math.sin(z)])
    if i == 120:
        break
    i = i + 1
# plt.subplot(3,1,3)
# plt.plot(T,predict)
ax2 = plt.gca()
ax2.set_xlim([t, k])

ax2.set_ylim([10, 150])
print(len(T), len(predict))
# print(predict)
#############################################################
for j in range(0, 120):
    if predict[j] < 40 and predict[j] > 0:
        print(Fore.BLACK + ' Water level low at:', time.ctime(T[j]), ',and the height is:', predict[j])
        print(Fore.RED + str(predict[j]))
flag = 0

print(Fore.BLACK + ' ')
for j in range(0, 120):
    if (predict[j] < 40 and predict[j] > 0):
        td = time.time() - T[j]
        day = datetime.datetime.fromtimestamp(T[j]).strftime("%A")

        while True:
            if day == 'Saturday' or day == 'Sunday':
                if int(time.time()) >= int(T[j] + td + 900):
                    print('The water level is predicted to be low at', time.ctime(T[j]),
                          ',but today is off,you need not turn on the motor.')
                    flag = 2
                    break

            elif int(time.time()) >= int(T[j] + td + 1):
                print('Warning!!Water Level is predicted to be low at,', time.ctime(T[j]), ' please fill your tank ')
                print('The height of tank is:', predict[j])
                recTime = time.time()
                print(' ')
                print('Please respond in 20 seconds! ')
                try:
                    for i in range(0, 20):
                        time.sleep(1)  # could use a backward counter to be preeety :)
                    print(' ')
                    print('Sorry!You have not responded.')
                    continue
                except KeyboardInterrupt:
                    input('Type anything:')
                    print('No more warnings!')
                    flag = 2
                    break
    if flag == 2:
        break
    if (predict[j] >= 100):
        td = time.time() - T[j]
        p = 0
        while True:
            p = p + 1
            if p == 1:
                print('The water level is high enough ', time.ctime(T[j]))
                print('The height of tank is:', predict[j])
                print(' ')
            if int(time.time()) == int(T[j] + td + 900):  # time.time()==T[j]-3600(when connected to the iOT device)
                print('The water level is high enough ', time.ctime(T[j]))
                print('The height of tank is:', predict[j])
                break
p2 = []
t2 = []

for i in range(24, 48):
    p2.append(predict[i])
    z = (datetime.datetime.fromtimestamp(int(T[i])).strftime('%H'))
    t2.append(z)
colors = []
for i in p2:
    if i < 40:
        colors.append('r')
    else:
        colors.append('b')
plt.subplot(2, 1, 2)
plt.bar(t2, p2, color=colors)
plt.savefig('tank13day2.jpg')
plt.show()
