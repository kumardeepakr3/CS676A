import numpy as np
import cv2
import sys
import math
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt


averageWindow = (5,5)
detectThreshold = 0
threshold = 1500
neighborhood_size = 5


# From https://gist.github.com/teechap/43988d7dc107ef79637d
def heatPlot(image):
	lenX, lenY = image.shape[1], image.shape[0]
	x = range(0, lenY)
	y = range(0, lenX)
	intensity = image.tolist()
	x, y = np.meshgrid(x,y)
	intensity = intensity = np.ma.masked_where(np.isnan(intensity), intensity)
	plt.pcolormesh(x, y, intensity.T)
	plt.colorbar()
	plt.show()
	x= raw_input()



def chooseProperTopCornerOfWindow(pixelX, pixelY, windowX, windowY, imageLenX, imageLenY):
	if(windowX > imageLenX or windowY > imageLenY):
		print "ERROR IN WINDOWSIZE"
	HalfWinX = windowX/2
	HalfWinY = windowY/2
	LeftWinX = HalfWinX
	TopWinY = HalfWinY
	RightWinX = windowX - LeftWinX -1
	BotWinY = windowY - TopWinY -1
	XTopCorner = pixelX - LeftWinX
	YTopCorner = pixelY - TopWinY
	if (pixelX - LeftWinX < 0):
		while (pixelX - LeftWinX < 0):
			LeftWinX = LeftWinX - 1
		XTopCorner = pixelX - LeftWinX
	if (pixelY - TopWinY < 0):
		while (pixelY - TopWinY < 0):
			TopWinY = TopWinY - 1
		YTopCorner = pixelY - TopWinY
	if (pixelX + LeftWinX > imageLenX-1):
		while (pixelX + LeftWinX > imageLenX-1):
			LeftWinX = LeftWinX -1
		XTopCorner = pixelX - (windowX-LeftWinX-1)
	if (pixelY + TopWinY > imageLenY-1):
		while (pixelY + TopWinY > imageLenY-1):
			TopWinY = TopWinY - 1
		YTopCorner = pixelY - (windowY - TopWinY-1)
	return (XTopCorner, YTopCorner)


def getGradientMatrix(grayImage):
	imageLenX, imageLenY = grayImage.shape[1], grayImage.shape[0]
	# grad = np.gradient(grayImage)
	# gradX = grad[1]
	# grayY = grad[0]
	gradX = cv2.Sobel(grayImage,cv2.CV_64F,1,0,ksize=5)  # x grad[1]
	gradY = cv2.Sobel(grayImage,cv2.CV_64F,0,1,ksize=5)  # y grad[0]
	gradMatrixForPixel = np.zeros(shape=(grayImage.shape[0], grayImage.shape[1], 4))
	for i in range(imageLenY):
		for j in range(imageLenX):
			gradMatrixForPixel[i][j][0] = (gradX[i][j])**2
			gradMatrixForPixel[i][j][3] = (gradY[i][j])**2
			gradMatrixForPixel[i][j][1] = gradMatrixForPixel[i][j][2] = gradX[i][j] * gradY[i][j]
	return gradMatrixForPixel


def getFValue(gradMatrix):
	global averageWindow
	windowX, windowY = averageWindow[0], averageWindow[1]
	lenX, lenY = gradMatrix.shape[1], gradMatrix.shape[0]
	FMatrix = np.zeros(shape=(lenY, lenX))
	for i in range(lenY):
		for j in range(lenX):
			(xtop, ytop) = chooseProperTopCornerOfWindow(j, i, windowX, windowY, lenX, lenY)
			windowSubMatrix = gradMatrix[np.ix_(range(ytop, ytop+windowY), range(xtop, xtop+windowX))]
			meanSubMatrix = np.mean(windowSubMatrix.reshape(windowX*windowY, 4), axis=0).reshape(2,2)
			det, trace = np.linalg.det(meanSubMatrix), np.trace(meanSubMatrix)
			if (trace == 0):
				FMatrix[i][j] = 0
				# print i, j
			else:
				FMatrix[i][j] = det/trace
	return FMatrix


if __name__ == "__main__":
	file_name = sys.argv[1]
	image = cv2.imread(file_name, 1)
	grayImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	print grayImage.shape
	gradMatrix = getGradientMatrix(grayImage)
	FValueMatrix = getFValue(gradMatrix)	
	# print FValueMatrix
	print FValueMatrix.shape
	# heatPlot(FValueMatrix)


	#Now Thresholding
	meanVal = np.mean(FValueMatrix)
	print meanVal
	newImg = FValueMatrix
	variance = np.var(FValueMatrix)
	# print "variance", variance
	print "MaxVal", np.amax(FValueMatrix)
	print "thresholdSum", str(meanVal+variance)
	threshold = meanVal+math.sqrt(variance)
	# cond1 = FValueMatrix <= 0.5*meanVal
	# FValueMatrix[cond1] = 0
	# cond2 = FValueMatrix > 0.5*meanVal
	# FValueMatrix[cond2] = 255
	# newImg = FValueMatrix.astype(np.uint8)

	#Now Finding LocalMaxima
	data_max = filters.maximum_filter(newImg, neighborhood_size)
	maxima = (newImg == data_max)
	data_min = filters.minimum_filter(newImg, neighborhood_size)
	diff = ((data_max - data_min) > threshold)
	maxima[diff == 0] = 0

	labeled, num_objects = ndimage.label(maxima)
	slices = ndimage.find_objects(labeled)
	x, y = [], []
	cornerMatrix = np.zeros(shape=newImg.shape)
	for dy,dx in slices:
		x_center = (dx.start + dx.stop - 1)/2
		x.append(x_center)
		y_center = (dy.start + dy.stop - 1)/2    
		y.append(y_center)
		cv2.circle(image, (x_center, y_center), 2, (0,0,255), -1)
		cornerMatrix[y_center][x_center] = 255

	cornerMatrix = cornerMatrix.astype(np.uint8)
	FValueMatrix = FValueMatrix.astype(np.uint8)

	cv2.imwrite('BikecornerMatrix.png', cornerMatrix)
	cv2.imwrite('BikefeaturePts.png', image)
	# cv2.imwrite('Graft2Img.png', newImg)
