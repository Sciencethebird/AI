import numpy as np 
import matplotlib.pyplot as plt
import os
import math
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

def avg_error():
    p = np.loadtxt('1D_Weights.txt', delimiter=',')
    error = 0
    d = 0
    print(p[0], p[1],p[2], p[3])
    for i in range(len(hp_test)):
        e = hp_test[i] - (p[1]*tr_test[i]
                         +p[2]*pop_test[i]
                         +p[3]*inc_test[i]
                         +p[0])
        d = d + e/hp_test[i]
        error = error + e**2
    error = math.sqrt(error/len(hp_test))
    d = d/len(hp_test)
    print('RMSD = ', error, ' dollar, error = ', 100*d, '%\n')

def avg_error_2d():
    p = np.loadtxt('2D_Weights.txt', delimiter=',')
    error = 0
    d = 0
    print(p)
    for i in range(len(hp_test)):
        e = hp_test[i] - (p[1]*tr_test[i]
                         +p[2]*pop_test[i]
                         +p[3]*inc_test[i]
                         +p[4]*tr_test[i]**2
                         +p[5]*pop_test[i]**2
                         +p[6]*inc_test[i]**2
                         +p[0])
        d = d + e/hp_test[i]
        error = error + e**2
    error = math.sqrt(error/len(hp_test))
    d = d/len(hp_test)
    print('RMSD = ', error, ' dollar, error = ', 100*d, '%\n')

def avg_error_3d():
    error = 0
    p = np.loadtxt('3D_Weights.txt', delimiter=',')
    print(p)
    error = 0
    d = 0
    for i in range(len(hp_test)):
        e = hp_test[i] - (p[0]+p[1]*total_room[i]
                              +p[2]*population[i]
                              +p[3]*median_income[i]
                              +p[4]*(total_room[i]**2)
                              +p[5]*(population[i]**2)
                              +p[6]*(median_income[i]**2)
                              +p[7]*(total_room[i]**3)
                              +p[8]*(population[i]**3)
                              +p[9]*(median_income[i]**3))
        d = d + e/hp_test[i]
        error = error + e**2
    error = math.sqrt(error/len(hp_test))
    d = d/len(hp_test)
    print('RMSD = ', error, ' dollar, error = ', 100*d, '%\n')

print('RMSD: ', np.average(hp_test),'\n')
avg_error_3d()
avg_error_2d()
avg_error()

