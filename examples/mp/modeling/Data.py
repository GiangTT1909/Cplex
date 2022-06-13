import csv
import numpy as np
datafile = open('data.csv')
datareader = csv.reader(datafile, delimiter=',')
data = []
for row in datareader:
    data.append(row)    

data = np.array(data)

data = np.array(data)
data =np.delete(data,0,1)
data =np.delete(data,0,1)
data =np.delete(data,0,1)

print(data)
