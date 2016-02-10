import numpy as np
import cv2
import sys
import math
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist




averageWindow = (5,5)
detectThreshold = 0
threshold = 1500
neighborhood_size = 10
patchSize = 16


img1FeatureList = []
img2FeatureList = []

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
	gradModDir = np.zeros(shape=(grayImage.shape[0], grayImage.shape[1], 2))
	for i in range(imageLenY):
		for j in range(imageLenX):
			Ix, Iy = gradX[i][j], gradY[i][j]
			gradMatrixForPixel[i][j][0] = Ix**2
			gradMatrixForPixel[i][j][3] = Iy**2
			gradMatrixForPixel[i][j][1] = gradMatrixForPixel[i][j][2] = Ix * Iy
			
			if(Ix == 0 and Iy == 0):
				gradModDir[i][j][1] = 0
				gradModDir[i][j][0] = 0
			elif(Ix ==0):
				gradModDir[i][j][1] = 90.0 if Iy>0 else 270.0
				gradModDir[i][j][0] = math.sqrt(Ix**2 + Iy**2)
			else:
				degVal = math.degrees(math.atan(Iy/Ix))
				if(Ix < 0):
					gradModDir[i][j][1] = degVal + 180.0
				elif(Iy < 0):
					gradModDir[i][j][1] = degVal + 360.0
				else:
					gradModDir[i][j][1] = degVal
				gradModDir[i][j][0] = math.sqrt(Ix**2 + Iy**2)

	return (gradMatrixForPixel, gradModDir)


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


def getHogBin(subRegionMatrix):
	lenX, lenY = subRegionMatrix.shape[1], subRegionMatrix.shape[0]
	h = [0] * 16   #histogram list
	diffAngle = 22.5
	rangeCenterList = [] # [11.25, 33.75, ... 348.75]
	for i in range(0,16):  # [0, 1, 2, 3, 4, .. 15]
		rangeCenterList.append(i*22.5+11.25)
	
	for i in range(lenX):
		for j in range(lenY):
			magnitude = subRegionMatrix[i][j][0]
			angle = subRegionMatrix[i][j][1]

			#BINNING PROCESS
			if(angle>=0 and angle<=11.25):
				h[0] += magnitude*((angle+11.25)/diffAngle)
				h[15] += magnitude*((11.25-angle)/diffAngle)
			elif (angle >=348.75):
				h[15] += magnitude*((371.25-angle)/diffAngle)  # 371.25 = 348.75+22.5
				h[0] += magnitude*((angle-348.75)/diffAngle)
			else:
				binNumber = int(angle/22.5)
				if ( int((angle+11.25)/22.5) > binNumber):
					rangeCenter = binNumber
				else:
					rangeCenter = binNumber - 1
				h[rangeCenter] += magnitude*((rangeCenterList[rangeCenter+1]-angle)/diffAngle)
				h[rangeCenter+1] += magnitude*((angle - rangeCenterList[rangeCenter])/diffAngle)
	# print h
	return h

			
def getHogVector(windowSubMatrix):
	global patchSize
	splitLen = patchSize/4
	rangeListX = range(0, patchSize, splitLen)
	rangeListY = range(0, patchSize, splitLen)
	hogBinList = []
	for i in [0, 1, 2, 3]:
		for j in [0, 1, 2, 3]:
			windowSubSubMatrix = windowSubMatrix[np.ix_(range(rangeListY[i], rangeListY[i]+splitLen), range(rangeListX[j], rangeListX[j]+splitLen))]
			hogBinList.extend(getHogBin(windowSubSubMatrix))
	return np.asarray(hogBinList)


def generateHOGVector(gradModDir, interestPointsList, imageNumber):
	global patchSize
	global img2FeatureList
	global img1FeatureList

	for interestPoint in interestPointsList:
		(xtop, ytop) = chooseProperTopCornerOfWindow(interestPoint[0], interestPoint[1], patchSize, patchSize, gradModDir.shape[1], gradModDir.shape[0])
		windowSubMatrix = gradModDir[np.ix_(range(ytop, ytop+patchSize), range(xtop, xtop+patchSize))]
		hogVectorOfWindow = getHogVector(windowSubMatrix)
		if imageNumber == 1:
			img1FeatureList.append((interestPoint, hogVectorOfWindow))
		else:
			img2FeatureList.append((interestPoint, hogVectorOfWindow))

def calculateSSD():
	global img1FeatureList
	global img2FeatureList
	img1FeatureVectorMatrix = np.zeros(shape=(256,))
	img2FeatureVectorMatrix = np.zeros(shape=(256,))
	
	for featureTuple in img1FeatureList:
		img1FeatureVectorMatrix = np.vstack([img1FeatureVectorMatrix, featureTuple[1]])
	img1FeatureVectorMatrix = np.delete(img1FeatureVectorMatrix, 0, 0) # Remove the first row of zeros

	for featureTuple in img2FeatureList:
		img2FeatureVectorMatrix = np.vstack([img2FeatureVectorMatrix, featureTuple[1]])
	img2FeatureVectorMatrix = np.delete(img2FeatureVectorMatrix, 0, 0) # Remove the first row of zeros

	# print img1FeatureVectorMatrix
	# print img1FeatureVectorMatrix.shape
	# print img2FeatureVectorMatrix.shape

	print "Calculating SSD Now"
	ssdMatrix = cdist(img1FeatureVectorMatrix, img2FeatureVectorMatrix)
	# print ssdMatrix
	print "SSD DONE"
	print "Shape of SSD Matrix=", ssdMatrix.shape
	return ssdMatrix




def generateAndDescribeInterestPoints(filename, image, imageNumber):
	global threshold
	grayImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	(gradMatrix, gradModDir) = getGradientMatrix(grayImage)
	FValueMatrix = getFValue(gradMatrix)
	meanVal = np.mean(FValueMatrix)
	newImg = FValueMatrix
	variance = np.var(FValueMatrix)
	threshold = meanVal+math.sqrt(variance)

	#Now Finding LocalMaxima
	data_max = filters.maximum_filter(newImg, neighborhood_size)
	maxima = (newImg == data_max)
	data_min = filters.minimum_filter(newImg, neighborhood_size)
	diff = ((data_max - data_min) > threshold)
	maxima[diff == 0] = 0

	labeled, num_objects = ndimage.label(maxima)
	slices = ndimage.find_objects(labeled)
	x, y = [], []
	interestPointsList = []
	cornerMatrix = np.zeros(shape=newImg.shape)
	for dy,dx in slices:
		x_center = (dx.start + dx.stop - 1)/2
		y_center = (dy.start + dy.stop - 1)/2    
		interestPointsList.append((x_center, y_center))
		cv2.circle(image, (x_center, y_center), 2, (0,0,255), -1)
		cornerMatrix[y_center][x_center] = 255
		
	print "Length of InterestPointList= ", len(interestPointsList)
	cornerMatrix = cornerMatrix.astype(np.uint8)
	FValueMatrix = FValueMatrix.astype(np.uint8)

	cv2.imwrite('Ans1_Corner_'+filename, cornerMatrix)
	cv2.imwrite('Ans1_Points_'+filename, image)

	generateHOGVector(gradModDir, interestPointsList, imageNumber)



#FOR SSD MATRIX colNum 
# def thresholdSSD(ssdMatrix, img1, img2):
# 	global img1FeatureList
# 	global img2FeatureList
# 	vis = np.concatenate((img1, img2), axis=1)

# 	minIndexForRow = np.argmin(ssdMatrix, axis=1) #Get index of minimum value of each row

# 	for i in range(len(minIndexForRow)):
# 		minIndex = minIndexForRow[i]
# 		(ximg1,yimg1) = img1FeatureList[i][0]
# 		(ximg2,yimg2) = img2FeatureList[minIndex][0]
# 		cv2.line(vis, (ximg1, yimg1), (ximg2+img1.shape[1], yimg2), (255,0,0))

# 	cv2.imwrite('combined.png', vis)


#FOR SSD MATRIX colNum 
def thresholdSSD(ssdMatrix, img1, img2):
	global img1FeatureList
	global img2FeatureList
	vis = np.concatenate((img1, img2), axis=1)

	minIndexForRow = np.argmin(ssdMatrix, axis=1) #Get index of minimum value of each row

	for i in range(len(minIndexForRow)):
		vis = np.concatenate((img1, img2), axis=1)
		minIndex = minIndexForRow[i]
		(ximg1,yimg1) = img1FeatureList[i][0]
		(ximg2,yimg2) = img2FeatureList[minIndex][0]
		cv2.line(vis, (ximg1, yimg1), (ximg2+img1.shape[1], yimg2), (255,0,0), 2)
		cv2.imwrite(str(i)+'combined.png', vis)






if __name__ == "__main__":
	filename1 = sys.argv[1]
	filename2 = sys.argv[2]

	image1 = cv2.imread(filename1, 1)
	image2 = cv2.imread(filename2, 1)
	generateAndDescribeInterestPoints(filename1, image1, 1)
	generateAndDescribeInterestPoints(filename2, image2, 2)
	ssdMatrix = calculateSSD()
	thresholdSSD(ssdMatrix, image1, image2)
	print ssdMatrix

	# print img1FeatureList[0][0]
	# print len(img1FeatureList[0][1])
	# grayImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

	# print grayImage.shape
	# (gradMatrix, gradModDir) = getGradientMatrix(grayImage)
	# print "GRADDIRDONE"
	# FValueMatrix = getFValue(gradMatrix)	
	# print FValueMatrix
	# print FValueMatrix.shape
	# heatPlot(FValueMatrix)


	#Now Thresholding
	# meanVal = np.mean(FValueMatrix)
	# print meanVal
	# newImg = FValueMatrix
	# variance = np.var(FValueMatrix)
	# print "variance", variance
	# print "MaxVal", np.amax(FValueMatrix)
	# print "thresholdSum", str(meanVal+variance)
	# threshold = meanVal+math.sqrt(variance)
	# cond1 = FValueMatrix <= 0.5*meanVal
	# FValueMatrix[cond1] = 0
	# cond2 = FValueMatrix > 0.5*meanVal
	# FValueMatrix[cond2] = 255
	# newImg = FValueMatrix.astype(np.uint8)

	#Now Finding LocalMaxima
	# data_max = filters.maximum_filter(newImg, neighborhood_size)
	# maxima = (newImg == data_max)
	# data_min = filters.minimum_filter(newImg, neighborhood_size)
	# diff = ((data_max - data_min) > threshold)
	# maxima[diff == 0] = 0

	# labeled, num_objects = ndimage.label(maxima)
	# slices = ndimage.find_objects(labeled)
	# x, y = [], []
	# cornerMatrix = np.zeros(shape=newImg.shape)
	# for dy,dx in slices:
	# 	x_center = (dx.start + dx.stop - 1)/2
	# 	x.append(x_center)
	# 	y_center = (dy.start + dy.stop - 1)/2    
	# 	y.append(y_center)
	# 	cv2.circle(image, (x_center, y_center), 2, (0,0,255), -1)
	# 	cornerMatrix[y_center][x_center] = 255

	# cornerMatrix = cornerMatrix.astype(np.uint8)
	# FValueMatrix = FValueMatrix.astype(np.uint8)

	# cv2.imwrite('ChesscornerMatrix.png', cornerMatrix)
	# cv2.imwrite('ChessfeaturePts.png', image)
	# cv2.imwrite('Graft2Img.png', newImg)
