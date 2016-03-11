#usage: python seamcarving.py img.png

import numpy as np
import cv2
import sys
import math
import scipy
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

NumSeamCarves = 100


def getGradientMatrix(grayImage):
	imageLenX, imageLenY = grayImage.shape[1], grayImage.shape[0]
	# grad = np.gradient(grayImage)
	# gradX = grad[1]
	# grayY = grad[0]
	gradX = cv2.Sobel(grayImage,cv2.CV_64F,1,0,ksize=5)  # x grad[1]
	gradY = cv2.Sobel(grayImage,cv2.CV_64F,0,1,ksize=5)  # y grad[0]
	gradMatrixForPixel = np.zeros(shape=(grayImage.shape[0], grayImage.shape[1]))
	for i in range(imageLenY):
		for j in range(imageLenX):
			gradMatrixForPixel[i][j] = (gradX[i][j])**2 + (gradY[i][j])**2

	return gradMatrixForPixel

def getMinEnergy(gradMatrix):
	minEnergyMatrix = np.zeros(shape=gradMatrix.shape)
	minEnergyMatrix[:] = -1
	imageLenX, imageLenY = gradMatrix.shape[1], gradMatrix.shape[0]
	for i in range(imageLenY):
		for j in range(imageLenX):
			if(i==0):
				minEnergyMatrix[i][j] = gradMatrix[i][j]
			elif(j==0):
				minEnergyMatrix[i][j] = gradMatrix[i][j] + min(minEnergyMatrix[i-1][j], minEnergyMatrix[i-1][j+1])
			elif(j==(imageLenX-1)):
				minEnergyMatrix[i][j] = gradMatrix[i][j] + min(minEnergyMatrix[i-1][j-1], minEnergyMatrix[i-1][j])
			else:
				minEnergyMatrix[i][j] = gradMatrix[i][j] + min(minEnergyMatrix[i-1][j-1], minEnergyMatrix[i-1][j], minEnergyMatrix[i-1][j+1])

	# print minEnergyMatrix
	return minEnergyMatrix
		


def getArrayToRemove(minEnergyMatrix):
	imageLenX, imageLenY = minEnergyMatrix.shape[1], minEnergyMatrix.shape[0]
	removalIndexArray = np.zeros((imageLenY,))
	m = imageLenY - 1
	removalIndexArray[m] = np.argmin(minEnergyMatrix[m])
	while (m>0):
		colIndex = removalIndexArray[m]
		# print colIndex
		if(colIndex == 0):
			removalIndexArray[m-1] = colIndex + np.argmin(np.array((minEnergyMatrix[m-1][colIndex], minEnergyMatrix[m-1][colIndex+1])))
		elif(colIndex == (imageLenX - 1)):
			removalIndexArray[m-1] = colIndex -1 + np.argmin(np.array((minEnergyMatrix[m-1][colIndex-1], minEnergyMatrix[m-1][colIndex])))
		else:
			removalIndexArray[m-1] = colIndex -1 + np.argmin(np.array((minEnergyMatrix[m-1][colIndex-1], minEnergyMatrix[m-1][colIndex], minEnergyMatrix[m-1][colIndex+1])))
		m = m-1
	# print removalIndexArray
	return removalIndexArray


def seamCarve(img, removalIndexArray):
	imageLenX, imageLenY = img.shape[1], img.shape[0]
	newImg = np.zeros((imageLenY, imageLenX-1, 3))
	m = n = 0
	i = j = 0
	while(i < imageLenY):
		remIndex = removalIndexArray[i]
		j = n = 0
		# print remIndex
		while(j < imageLenX):
			if j == remIndex:
				j=j+1
			else:
				newImg[m][n] = img[i][j]
				j = j+1
				n = n+1
			# print (i,j)
		# print  (j, n)
		i= i+1
		m = m+1
	return newImg.astype(np.uint8)



def performSeamCarving(img):
	img = cv2.imread(imgname, 1)
	global NumSeamCarves
	for i in range(NumSeamCarves):
		print "Iteration "+str(i+1)+ "/" +str(NumSeamCarves)
		grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		gradMatrix = getGradientMatrix(grayImage)
		minEnergyMatrix = getMinEnergy(gradMatrix)
		removalIndexArray = getArrayToRemove(minEnergyMatrix)
		newImg = seamCarve(img, removalIndexArray)
		img = newImg
	cv2.imwrite('carved.png', newImg)

if __name__ == "__main__":
	imgname = sys.argv[1]
	img = cv2.imread(imgname, 1)
	performSeamCarving(img)
