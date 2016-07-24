---
layout: post
title: Identify up/down trends in historical stock data
image: /public/img/gmt.jpg
color: '#949667'
tags: [Trading]
---

## Identify up/down trends in historical stock data
* This script is part of a research project about applying machine learning in trading algorithm.
* Identifying up/down trends is necessary for collecting training data in this project, I hope it could help with some other projects.
* I used quadratic polynomial regression to smooth the historical data and identify the trend.
* The result is not bad, but still needs to be improved, and since I used regression, the result may not be highly precise.


```python
%matplotlib inline 
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tseries.offsets import *
import numpy as np
import scipy
import pandas.io.data as web
import datetime 
import seaborn as sns
import pylab
from pandas.tseries.offsets import *
pylab.rcParams['figure.figsize'] = (10.0, 8.0)
sns.set_style("darkgrid")
```


```python
#read historical data from google finance
start = datetime.datetime(2014, 1, 1)
end = datetime.datetime(2017, 1, 27)
history=web.DataReader('JD','google',start,end)['Close']
```

## polynomial regression
Here I used `np.polyfit` to do the quadratic polynomial regression, which returns three coefficients, then use `np.poly1d` to build a formula based on the coefficients.


```python
#get the formula
f=np.polyfit(x=np.arange(20),y=history[0:20].values,deg=2)
print("coefficients: ",f)
f= np.poly1d(f)

#use formula to calculate regression line
xnew = np.arange(0, 19, 0.1)
ynew=f(np.arange(20))

plt.plot(np.arange(20),history[0:20].values,'o', np.arange(20),ynew, '-')
```

    coefficients:  [ -0.03872693   1.03380417  20.59913636]
    









![png](https://raw.githubusercontent.com/YichengPu/YIchengPu.github.io/master/public/img/id_trends/1.png)


## Identify trends in one period
After we get the regreesion line, since the first coefficient is negative, we know in this period the up trend comes before the down trend, then by finding the maximum, we split this period into two trends.(red for up and green for down)


```python
#find the maximum
p=([i for i, j in enumerate(history[0:20]) if j == max(history[0:20])])[0]

plt.plot(np.arange(20),history[0:20].values,'o', np.arange(20),ynew, '-')
plt.axvspan(0, p+1, alpha=0.4, color='red')
plt.axvspan(p+1, 19, alpha=0.4, color='green')
```









![png](https://raw.githubusercontent.com/YichengPu/YIchengPu.github.io/master/public/img/id_trends/2.png)


## Main part
Here is the main part.
* First we need to set a window size to split the historical data into smaller periods.
* Then we can treat each small periods like we did above.
* We store the results in "trend" list. 


```python
#set up the window size
min_window=5

#Initialize,q_fit is used to store the regression line.
d=min_window
prev_d=0
trend=np.zeros(len(history))
q_fit=np.zeros(len(history))

#Loop through all small periods
while d<(len(history)-min_window):
    # polynomial regression
    x=np.arange(min_window)
    y=history[prev_d:d].values
    fit=np.polyfit(x=x,y=y,deg=2)
    formula= np.poly1d(fit)
    q_fit[prev_d:d]=formula(x)
    
    #split
    if fit[0] >0:
        point=([i for i, j in enumerate(y) if j == min(y)])[0]+prev_d
        trend[prev_d:point+1]=-1
        trend[point+1:d]=1
    else:
        point=([i for i, j in enumerate(y) if j == max(y)])[0]+prev_d
        trend[prev_d:point+1]=1
        trend[point+1:d]=-1        
    prev_d=d
    d+=min_window

#This part does the same thing as the procedure in loop for the very small end period (less than window size) 
d-=min_window
remain_d=len(history)-d
x=np.arange(remain_d)
y=history[d:len(history)].values
fit=np.polyfit(x=x,y=y,deg=2)
formula= np.poly1d(fit)
q_fit[d:len(history)]=formula(x)
if fit[0] >0:
    point=([i for i, j in enumerate(y) if j == min(y)])[0]+prev_d
    trend[d:point+1]=-1
    trend[point+1:len(history)]=1
else:
    point=([i for i, j in enumerate(y) if j == max(y)])[0]+prev_d
    trend[d:point+1]=1
    trend[point+1:len(history)]=-1   
```

## Plot


```python
# we plot the historical price, regression line and trend line.
plt.plot(q_fit[0:100])
plt.plot(history.values[0:100],color='blue')
plt.plot(trend[0:100])

# Fill the vertical area in each trend
ps=-1
lp=0
for i, j in enumerate(trend):
    if i<100:
        if i<len(trend)-1:
            if ps==-1 and j == 1:
                    plt.axvspan(lp, i-1, alpha=0.4, color='green')
                    lp=i-1
                    ps=1
            if ps==1 and j==-1:
                    plt.axvspan(lp, i-1, alpha=0.4, color='red')
                    lp=i-1
                    ps=-1
        else:
                if ps==1:
                    plt.axvspan(lp, i-1, alpha=0.4, color='green')
                if ps==-1:
                    plt.axvspan(lp, i-1, alpha=0.4, color='red')

```


![png](https://raw.githubusercontent.com/YichengPu/YIchengPu.github.io/master/public/img/id_trends/3.png)






