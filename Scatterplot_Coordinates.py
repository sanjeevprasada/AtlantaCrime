import csv
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as sk

# Open the CSV and store it in a list
coordinates = []
with open("Coordinates.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        coordinates.append(row)

# Initialize x and y vectors
x = np.zeros(shape=(len(coordinates), 1))
y = np.zeros(shape=(len(coordinates), 1))
# Store coordinates
for i in np.arange(len(coordinates)):
    x[i] = coordinates[i][1]
    y[i] = coordinates[i][2] 

# Normalize the x and y values
x = x - x.min(axis=0)
y = y - y.min(axis=0)
x_max = x.max(axis=0)
y_max = y.max(axis=0)
norm_x = x/x_max
norm_y = y/y_max

# Size of the points we use to draw on the scatterplot
area = np.pi*0.01

# Draw scatterplot
plt.scatter(norm_x, norm_y, s=area, c='black', alpha=0.5)
plt.title('Scatter plot pythonspot.com')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('Atlanta_Crime.png')
plt.show()