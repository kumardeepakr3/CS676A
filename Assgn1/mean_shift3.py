import numpy as np
import cv2
import sys
import math
import scipy.ndimage
import copy


ModeDictionary = {}
basin_list = []
sr = 2



def generate_5d(image):
	(row,column,x)=image.shape
	final_matrix = np.zeros(shape=(row*column,5))
	iterator=0
	for i in xrange(0,row):
		for j in xrange(0,column):
			final_matrix[iterator]=[i,j,image[i][j][0],image[i][j][1],image[i][j][2]]
			iterator = iterator + 1
	return (final_matrix,row,column)


def gaussian(bandwidth_loc,bandwidth_col,distance_loc,distance_col):
	val = 1/(bandwidth_col*bandwidth_loc*2*math.pi)*np.exp(-0.5 *( ( distance_loc/(bandwidth_loc**2)+distance_col/(bandwidth_col**2))  ))
	#print val
	return val


def flat(bandwidth_loc, bandwidth_col, distance_loc, distance_col):
	if distance_col < bandwidth_col and distance_loc < bandwidth_loc:
		return 1
	else:
		return 0


def distance(array1,array2):
	array = array1 - array2
	distance_col=array[2]**2 + array[3]**2 + array[4]**2
	# print array[2],array[3],array[4]
	distance_loc = array[0]**2 + array[1]**2
	#print (distance_loc,distance_col)
	return (distance_loc,distance_col)


def KDE(image_5d,bandwidth_col,bandwidth_loc,row,column):
	size = row*column
	kde = np.zeros(shape=(row,column))
	for i in xrange(0,size):
		for x in xrange(0,size):
			(distance_loc,distance_col)=distance(image_5d[row],image_5d[x])
			kde[image_5d[row][0]][image_5d[row][1]] += gaussian(bandwidth_loc,bandwidth_col,distance_loc,distance_col) 
		# print i
	return kde


def KDE2(image_5d,bandwidth_col,bandwidth_loc,row,column):
	size = row*column
	kde = np.zeros(shape=(row,column))
	for i in xrange(0,size):
		for x in xrange(0,size):
			(distance_loc,distance_col)=distance(image_5d[row],image_5d[x])
			kde[image_5d[row][0]][image_5d[row][1]] += flat(bandwidth_loc,bandwidth_col,distance_loc,distance_col) 
		print i
	return kde

def diff(a, b):
	if(a>b):
		return a-b
	else:
		return b-a


def KDE3(img,bandwidth_col,bandwidth_loc,row,column):
	imageLenX = img.shape[0]
	imageLenY = img.shape[1]

	kde = np.zeros(shape=(imageLenX, imageLenY))
	windowX = int(math.ceil(2*math.sqrt(bandwidth_loc) + 1))
	windowY = windowX
	#print type(windowX), type(windowY)

	distance_loc = 0
	distance_col = 0

	for i in xrange(0, imageLenX):
		for j in xrange(0, imageLenY):
			(xtop, ytop) = chooseProperTopCornerOfWindow(i, j, windowX, windowY, imageLenX, imageLenY)
			#print xtop, ytop 
			windowSubMatrix = kde[np.ix_(range(xtop, xtop+windowX), range(ytop, ytop+windowY))]
			for x in xrange(0, windowSubMatrix.shape[0]):
				for y in xrange(0, windowSubMatrix.shape[1]):
					distance_loc = (i-x-xtop)**2 + (j-y-ytop)**2
					distance_col = diff(img[i][j][0] , img[xtop+x][ytop+y][0])**2 + \
									diff(img[i][j][1] , img[xtop+x][ytop+y][1])**2 + \
									diff(img[i][j][2] , img[xtop+x][ytop+y][2])**2
					kde[i][j] += flat(bandwidth_loc, bandwidth_col, distance_loc, distance_col)
			#print kde[i][j]	
		print i
	return kde

def KDEGauss(img,bandwidth_col,bandwidth_loc,row,column):
	imageLenX = img.shape[0]
	imageLenY = img.shape[1]

	kde = np.zeros(shape=(imageLenX, imageLenY))
	windowX = int(math.ceil(2*math.sqrt(bandwidth_loc) + 1))
	windowY = windowX
	#print type(windowX), type(windowY)

	distance_loc = 0
	distance_col = 0

	for i in xrange(0, imageLenX):
		for j in xrange(0, imageLenY):
			(xtop, ytop) = chooseProperTopCornerOfWindow(i, j, windowX, windowY, imageLenX, imageLenY)
			#print xtop, ytop 
			windowSubMatrix = kde[np.ix_(range(xtop, xtop+windowX), range(ytop, ytop+windowY))]
			for x in xrange(0, windowSubMatrix.shape[0]):
				for y in xrange(0, windowSubMatrix.shape[1]):
					distance_loc = (i-x-xtop)**2 + (j-y-ytop)**2
					distance_col = diff(img[i][j][0] , img[xtop+x][ytop+y][0])**2 + \
									diff(img[i][j][1] , img[xtop+x][ytop+y][1])**2 + \
									diff(img[i][j][2] , img[xtop+x][ytop+y][2])**2
					kde[i][j] += gaussian(bandwidth_loc, bandwidth_col, distance_loc, distance_col)
			#print kde[i][j]	
		print i
	return kde


def getMeaninRegion(windowSubMatrix):
	COM = scipy.ndimage.measurements.center_of_mass(windowSubMatrix)
	#print windowSubMatrix
	#print "\n\n\n"
	return (int(math.floor(COM[0])), int(math.floor(COM[1])))


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


def updateDictionary(meanX, meanY, pixelX, pixelY):
	global ModeDictionary
	if (meanX,meanY) in ModeDictionary:
		ModeDictionary[(meanX,meanY)].append((pixelX,pixelY))
	else:
		ModeDictionary[(meanX,meanY)] = [(pixelX,pixelY)]

	

def can_merge(i, j,img,distance_loc,distance_col):
	global basin_list
	# global sr
	basin1 = basin_list[i]
	basin2 = basin_list[j]
	x1 = basin1[0][0]
	y1 = basin1[0][1]
	x2 = basin2[0][0]
	y2 = basin2[0][1]
	space_dist_sq = (diff(x1,x2)**2 + diff(y1,y2)**2)
	if space_dist_sq <= distance_loc:
		return True
	c1 = img[x1, y1]
	c2 = img[x2, y2]
	rr = 8
	color_dist_sq = diff(c1[0],c2[0])**2 + diff(c1[1],c2[1])**2 + diff(c1[2],c2[2])**2
	if color_dist_sq <= distance_col:
		return True
	return False

def merge(i, j, img, kde):
 	global ModeDictionary
 	global basin_list
 	basin1 = basin_list[i]
 	basin2 = basin_list[j]
 	x1 = basin1[0][0]
 	y1 = basin1[0][1]
 	x2 = basin2[0][0]
 	y2 = basin2[0][1]
 	c1 = img[x1][y1]
 	c2 = img[x2][y2]
 	pdf1 = kde[x1][y1]
 	pdf2 = kde[x2][y2]
 	if pdf1 >= pdf2:
 		mode_x = x1
 		mode_y = y1
 	else:
 		mode_x = x2
 		mode_y = y2
 	merged_basin = ((mode_x, mode_y), basin1[1] + basin2[1])
 	basin_list = basin_list[:i] + [merged_basin] + basin_list[i+1:j] + basin_list[j+1:]

def shift_pixel(image,kde,window_size):
	global ModeDictionary
	global basin_list
	windowX = window_size[0]
	windowY = window_size[1]
	imageLenX = image.shape[0]
	imageLenY = image.shape[1]
	pixelClusterLocn = np.zeros(shape=(imageLenX, imageLenY, 2))
	clusterLocationMatrix = np.zeros(shape=image.shape)
	start_pixelX = 0
	start_pixelY = 0

	retImage = np.zeros(shape=(imageLenX, imageLenY, 3))

	for i in xrange(0, imageLenX):
		print i
		for j in xrange(0, imageLenY):
			ReachedMean = False
			PrevMean = (i,j)
			while(not ReachedMean):
				#print i,j
				(xtop, ytop) = chooseProperTopCornerOfWindow(PrevMean[0],PrevMean[1], windowX, windowY, imageLenX, imageLenY)
				#print (xtop, xtop+windowX),(ytop, ytop+windowY)
				windowSubMatrix = kde[np.ix_(range(xtop, xtop+windowX), range(ytop, ytop+windowY))]
				if(np.count_nonzero(windowSubMatrix) == 0):
					(meanX, meanY) = PrevMean
				else:
					(meanX, meanY) = getMeaninRegion(windowSubMatrix)
					meanX = meanX + xtop
					meanY = meanY + ytop
				#print PrevMean
				#print (meanX, meanY)
				#print
				# if(meanX, meanY) == PrevMean:
				# 	ReachedMean = True
				if ((abs(PrevMean[0]-meanX) + abs(PrevMean[1]-meanY)) < 5):
					ReachedMean = True
				else:
					PrevMean = (meanX, meanY)
			# if(image[meanX][meanY] > 50):
			# 	pass
			# 	print i,j
			# 	print PrevMean
			# 	print image[meanX][meanY]
			# 	print retImage[i][j]
			# 	#k = raw_input()
			# else:
			# 	pass
			updateDictionary(meanX, meanY, i, j)
			pixelClusterLocn[i][j] = [meanX, meanY]
			clusterLocationMatrix[meanX][meanY] +=1
			retImage[i][j] = image[meanX][meanY]
			# if [meanX,meanY] not in basin_list:
			# 	basin_list.append([meanX,meanY])
	# print type(retImage)
	# print type(retImage[0][0])

	print pixelClusterLocn
	print clusterLocationMatrix
	return retImage.astype(np.uint8)




if __name__ == "__main__":
	file_name = sys.argv[1]
	bandwidth_col = 100
	bandwidth_loc = 33
	row = 256
	column =256
	image = cv2.imread(file_name, 1)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2Luv)
	global basin_list
	global ModeDictionary
	# image = np.zeros(shape=(25,25,3))
	# image[15][15][0]=255
	# image[15][15][1]=255
	# image[15][15][2]=255
	# image = cv2.imread('42049.jpg',1)
	# cv2.imwrite('mess.jpg',image)
	# cv2.imshow('abc.jpg',image)
	# cv2.waitKey(0)

	#(image_5d,row,column)=generate_5d(image)
	#print "Generated 5D"
	kde=KDEGauss(image,bandwidth_col,bandwidth_loc,row,column)
	retImg = shift_pixel(image, kde, (6,6))
	basin_list = sorted(ModeDictionary.iteritems())
	for i in range (0, len(basin_list)):
		for j in range(i+1, len(basin_list)):
			if j >= len(basin_list):
				break # since basin_list is being changed
			if can_merge(i, j,image,bandwidth_loc,bandwidth_col):
				merge(i, j,image,kde)

	new_img = copy.deepcopy(image)
	# form segmented image
	for i in range(0, len(basin_list)):
		b = int (image[basin_list[i][0]][0])
		g = int (image[basin_list[i][0]][1])
		r = int (image[basin_list[i][0]][2])
		for (x, y) in basin_list[i][1]:
			new_img[x][y] = [b, g, r]
	# new_rgb_img = cv2.cvtColor(new_img, cv2.cv.CV_Luv2BGR)
	new_rgb_img= new_img.astype(np.uint8)
	new_rgb_img = cv2.cvtColor(new_rgb_img, cv2.COLOR_Luv2BGR)
	cv2.imwrite("LUVTry.png", new_rgb_img)




#retImg2 = np.zeros(shape=(img.shape[0], img.shape[1]))
	print image

	print "Done KDE"
	print (len(basin_list))
	# retImg = shift_pixel(image, kde, (4,4))
	# cv2.imshow('ImgRet', retImg)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	# cv2.imwrite('Building2020.jpg', retImg)
	# print kde

	# shift_pixel(image,kde,window_size)
	
