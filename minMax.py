import numpy as np
import cv2
import sys
import math
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt


threshold = 150
neighborhood_size = 5



if __name__ == "__main__":

	fname = sys.argv[1]
	newImg = scipy.misc.imread(fname)
	data_max = filters.maximum_filter(newImg, neighborhood_size)
	maxima = (newImg == data_max)
	data_min = filters.minimum_filter(newImg, neighborhood_size)
	diff = ((data_max - data_min) > threshold)
	maxima[diff == 0] = 0

	labeled, num_objects = ndimage.label(maxima)
	slices = ndimage.find_objects(labeled)
	x, y = [], []
	for dy,dx in slices:
		x_center = (dx.start + dx.stop - 1)/2
		x.append(x_center)
		y_center = (dy.start + dy.stop - 1)/2    
		y.append(y_center)

	plt.imshow(newImg)
	# plt.savefig('data.png', bbox_inches = 'tight')

	plt.autoscale(False)
	plt.plot(x,y, 'ro')
	plt.savefig('result.png', bbox_inches = 'tight')
