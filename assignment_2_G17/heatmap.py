import matplotlib.pyplot as plt
import numpy as np

#here's our data to plot, all normal Python lists
x = [1, 2, 3, 4, 5, 6, 7]
y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

intensity = [
    [5, 10, 15, 20, 25, np.nan],
    [30, 35, 40, 45, 50, np.nan],
    [55, 60, 65, 70, 75, np.nan],
    [80, 85, 90, 95, 100, np.nan],
    [105, 110, 115, 120, 125, np.nan],
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
]

#setup the 2D grid with Numpy
x, y = np.meshgrid(x, y)

#convert intensity (list of lists) to a numpy array for plotting
#ma.masked_where doesnt keeps spaces blank if there's no data for them (i.e. type np.nan)
intensity = np.ma.masked_where(np.isnan(intensity), intensity)

#now just plug the data into pcolormesh, it's that easy!
plt.pcolormesh(x, y, intensity.T)
plt.colorbar() #need a colorbar to show the intensity scale
plt.show() #boom