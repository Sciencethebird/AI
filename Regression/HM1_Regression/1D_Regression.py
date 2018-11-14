import numpy as np 
import matplotlib.pyplot as plt
import os
import csv

total_room, population, median_income, median_house_price = np.loadtxt('data.csv', delimiter = ',', unpack = True)



tr_train = total_room[0:16513]
pop_train = population[0:16513]
inc_train = median_income[0:16513]
hp_train = median_house_price[0:16513]

tr_test = total_room[16513:]
pop_test = population[16513:]
inc_test = median_income[16513:]
hp_test = median_house_price[16513:]

def avg_error(w, b):
    error = 0
    for i in range(len(hp_test)):
        e = abs(hp_test[i] - (w*inc_test[i]+b))/hp_test[i]
        #print(e)
        error = error + e
    error = error/len(hp_test)
    print(w,' ,', b,' average error =', error*100, '%')

#Basic Practice of one feature
b = 1
w = 0
lr = 0.000001
iteration = 100
b_history =[]
w_history =[]
for i in range(iteration):
    b_grad = 0.0
    w_grad = 0.0
    if i%100 is 0:
        print(int(i/iteration*100),'% done')
    for n in range (len(median_house_price)):
        #print(population[n])
        b_grad = b_grad - 2.0*(median_house_price[n]
                               -b-w*median_income[n])*1.0
        w_grad = w_grad - 2.0*(median_house_price[n]
                               -b-w*median_income[n])*median_income[n]

    b = b - lr * b_grad
    w = w - lr * w_grad
    b_history.append(b)
    w_history.append(w)

avg_error(w, b)
plt.plot(np.arange(len(b_history)), b_history, label = 'w')
plt.ylim(-50000, 50000)
plt.xlabel('iteration')
plt.ylabel('w')
plt.show()

plt.scatter(np.arange(len(hp_test)),hp_test, label = 'test_data', color = 'b')
prdic = []
for i in range(len(inc_test)):
    prdic.append(w*inc_test[i]+b)
print(prdic)
plt.plot(np.arange(len(hp_test)),prdic, label = 'prediction', color = 'orange')
plt.ylim(-500000, 500000)
plt.xlabel('i')
plt.ylabel('price')
plt.show()