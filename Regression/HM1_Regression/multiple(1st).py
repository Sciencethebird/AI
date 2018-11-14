
import numpy as np 
import matplotlib.pyplot as plt
import os
import csv

total_room, population, median_income, median_house_price = np.loadtxt('data.csv', delimiter = ',', unpack = True)


room_train = total_room[0:16513]
popu_train = population[0:16513]
income_train = median_income[0:16513]
price_train = median_house_price[0:16513]

room_test = total_room[16513:]
popu_test = population[16513:]
income_test = median_income[16513:]
price_test = median_house_price[16513:]


#Basic Practice of one feature

b = 0
w1 = 0 #total room
w2 = 0 #population
w3 = 0 #income

lrb = 0.000001
lr1 = 0.000000000001
lr2 = 0.000000000001
lr3 = 0.000001

iteration = 3000

b_history =[]
w1_history =[]
w2_history =[]
w3_history =[]


for i in range(iteration):

    b_grad = 0.0
    w1_grad = 0.0
    w2_grad = 0.0
    w3_grad = 0.0

    if i%100 is 0:
        print(int(i/iteration*100),'% done')

    for n in range (len(median_house_price)):
        temp = 2.0*(median_house_price[n]
                               -b
                               -w1*total_room[n]
                               -w2*population[n]
                               -w3*median_income[n])

        b_grad = b_grad - temp * 1.0
        w1_grad = w1_grad - temp * total_room[n]
        w2_grad = w2_grad - temp * population[n]
        w3_grad = w3_grad - temp * median_income[n]
        
    b = b - lrb * b_grad
    w1 = w1 - lr1 * w1_grad
    w2 = w2 - lr2 * w2_grad
    w3 = w3 - lr3 * w3_grad
    
    b_history.append(b) 
    w1_history.append(w1)
    w2_history.append(w2)
    w3_history.append(w3)
  


with open('1D_Weights.txt', 'w') as file:
    weights = [b, w1, w2, w3]
    out = str(weights)[1:-1]
    file.write(out)

with open('1D_Weights_history.txt', 'w') as file:
    file.write(str(b_history)[1:-1]+'\n')
    file.write(str(w1_history)[1:-1]+'\n')
    file.write(str(w2_history)[1:-1]+'\n')
    file.write(str(w3_history)[1:-1]+'\n')
#hp.avg_error(w, b, inc_test, hp_test)

plt.plot(np.arange(len(b_history)), b_history, label = 'b')
#plt.plot(np.arange(len(b_history)), w2_history, label = 'w2')
plt.legend()
plt.xlabel('iteration')
plt.ylabel('b')
plt.show()

plt.plot(np.arange(len(b_history)), w1_history, label = 'w1')
#plt.plot(np.arange(len(b_history)), w2_history, label = 'w2')
plt.legend()
plt.xlabel('iteration')
plt.ylabel('w1')
plt.show()

plt.plot(np.arange(len(b_history)), w2_history, label = 'w2')
#plt.plot(np.arange(len(b_history)), w2_history, label = 'w2')
plt.legend()
plt.xlabel('iteration')
plt.ylabel('w2')
plt.show()

plt.plot(np.arange(len(b_history)), w3_history, label = 'w3')
#plt.plot(np.arange(len(b_history)), w2_history, label = 'w2')
plt.legend()
plt.xlabel('iteration')
plt.ylabel('w3')
plt.show()
'''
plt.scatter(np.arange(len(hp_test)),hp_test, label = 'test_data', color = 'b')
prdic = []
for i in range(len(inc_test)):
    prdic.append(w*inc_test[i]+b)
print(prdic)
plt.plot(np.arange(len(hp_test)),prdic, label = 'prediction', color = 'orange')
plt.ylim(-5000, 5000)
plt.xlabel('income')
plt.ylabel('price')
plt.show()
'''