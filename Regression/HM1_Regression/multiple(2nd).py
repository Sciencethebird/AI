

import numpy as np 
import matplotlib.pyplot as plt
import os
import csv

total_room, population, median_income, median_house_price = np.loadtxt('data.csv', delimiter = ',', unpack = True)


x1 = total_room[0:16513]
x2 = population[0:16513]
x3 = median_income[0:16513]
y = median_house_price[0:16513]

room_test = total_room[16513:]
popu_test = population[16513:]
income_test = median_income[16513:]
price_test = median_house_price[16513:]


b = 0.01
w1 = 0.01 #total room
w2 = 0.01 #population
w3 = 0.01 #income
w4 = 0.0
w5 = 0.0
w6 = 0.0

lrb = 0.000001
lr1 = 0.000000000001
lr2 = 0.000000000001
lr3 = 0.000001
lr4 = 0.00000000000000000001 
lr5 = 0.00000000000000000001
lr6 = 0.00000001

iteration = 1500

b_history =[]
w1_history =[]
w2_history =[]
w3_history =[]
w4_history =[]
w5_history =[]
w6_history =[]


for i in range(iteration):

    b_grad = 0.0
    w1_grad = 0.0
    w2_grad = 0.0
    w3_grad = 0.0
    w4_grad = 0.0
    w5_grad = 0.0
    w6_grad = 0.0




    if i%10 is 0:
        print(int(i/iteration*100),'% done')


    for n in range (len(y)):
      
        temp = 2.0 * (y[n] - b -w1*x1[n]
                               -w2*x2[n]
                               -w3*x3[n]
                               -w4*(x1[n]**2)
                               -w5*(x2[n]**2)
                               -w6*(x3[n]**2))

        b_grad = b_grad - temp * 1.0      
        w1_grad = w1_grad - temp * x1[n]                                      
        w2_grad = w2_grad - temp * x2[n]
        w3_grad = w3_grad - temp * x3[n]                                                            
        w4_grad = w4_grad - temp * (x1[n]**2)                     
        w5_grad = w5_grad - temp * (x2[n]**2)
        w6_grad = w6_grad - temp * (x3[n]**2)
        
    b = b - lrb * b_grad
    w1 = w1 - lr1 * w1_grad
    w2 = w2 - lr2 * w2_grad
    w3 = w3 - lr3 * w3_grad
    w4 = w4 - lr4 * w4_grad
    w5 = w5 - lr5 * w5_grad
    w6 = w6 - lr6 * w6_grad
    
    b_history.append(b) 
    w1_history.append(w1)
    w2_history.append(w2)
    w3_history.append(w3)
    w4_history.append(w4)
    w5_history.append(w5)
    w6_history.append(w6)
  


with open('2D_Weights.txt', 'w') as file:
    weights = [b, w1, w2, w3, w4, w5, w6]
    file.write(str(weights)[1:-1])

def plot_weight(weight, name):
    plt.plot(np.arange(len(b_history)), weight, label = name)
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel(name)
    plt.show()

plot_weight(b_history, 'b')
plot_weight(w1_history, 'w1')
plot_weight(w2_history, 'w2')
plot_weight(w3_history, 'w3')
plot_weight(w4_history, 'w4')
plot_weight(w5_history, 'w5')
plot_weight(w6_history, 'w6')

