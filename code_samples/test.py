import numpy as np
import matplotlib.pyplot as plt

def add_one(x):
	return x + 1 
	
x = np.linspace(0,10)
y= np.sin(x)

plt.plot(x,y)