#coding=utf-8
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
import matplotlib

#calculate differenced value of each element
def differenced(X, interval=1):
    diffed=list()
    for i in range(interval, len(X)):
        value= X[i] - X[i-interval]
        diffed.append(value)
    return np.array(diffed)

#invert differenced to original
def invert_differenced(history , yhat , interval= 1):
    return yhat + history[-interval]

#load dataset
print("loading file...")
data=pd.read_csv("drgrace_13th.csv")
print("load successful")
datavalue= data['value']

#separate dataset into two groups, one for training and another for testing
test_size=int(len(datavalue) * 0.7)
train,test_tmp= datavalue[:1000],datavalue[test_size:test_size+80000]
test=[x for x in test_tmp]

#the cycle of the dataset: one month = 30 days * 24h
interval= 100

diffed= differenced(train,interval)

print("modeling...")
model=ARIMA(diffed,order=(0,1,1))
model_fit=model.fit(disp=0)
steps=80000
forcast=model_fit.forecast(steps=steps)[0]

#invert the forcast result
history= [x for x in train]
for i in range(len(forcast)):
    forcast[i]= invert_differenced(history,forcast[i],interval)
    history.append(forcast[i])
    print("step1...", i)
yforcast=[x for wx in train]
for i in range(steps):
    yforcast.append(forcast[i])
    print("step2...", i)

#dif_test_forcast=test - forcast
#for i in range(len(test)):
   # print 'test: ',test[i],' forcast: ',forcast[i],' dif: ',dif_test_forcast[i],' percent: %.2f%%' % (float(dif_test_forcast[i]) /float(test[i]) * 100)

#plt.plot(train+50)
#axis = range(steps)

print("len(steps)", steps)
plt.figure(1)
plt.plot(test,color='blue')#test plot
plt.plot(forcast,color='red')      #predict plot
#plt.plot(test - forcast,color='black')#(test - predict) plot
plt.show()


