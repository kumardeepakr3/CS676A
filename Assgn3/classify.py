import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from collections import defaultdict
import pickle
import math


k = 10
heightVocabTree = 4
t = 400 # Max number of SIFT features to extract from an image
total_num_imgs = 0

qiSumTotal = 1
kMeansClustererDumpPath = "/home/deepak/Downloads/dump/"
pathOfImageFolder = "/home/deepak/Downloads/ukbenchSampleSet700/"
clustererFileName = "ClustererDict_ukbench00000_700_10_4.pckl"
kMeansClusterDict = {}
testImgPath = "/home/deepak/Downloads/ukbenchSampleSet500/ukbench00466.jpg"
distanceDict = {}




def simpleVoting():
	global kMeansClusterDict
	global distanceDict
	global heightVocabTree
	global allDatabaseImgList

	kMeansClusterDict = pickle.load(open(kMeansClustererDumpPath+clustererFileName, 'r'))
	img = cv2.imread(testImgPath, 1)
	desc = getSiftDescriptorOfImage(img)
	print desc.shape

	# rootNode = kMeansClusterDict[tuple([0])]
	# allDatabaseImgList = rootNode[2].keys()
	for img in allDatabaseImgList:
		distanceDict[img] = 0
	for vector in desc:
		currHeightList = [0]
		while(len(currHeightList)!=heightVocabTree+1):
			currNode = kMeansClusterDict[tuple(currHeightList)]
			curr_clusterer = currNode[0]
			dVector = currNode[2]
			for imgname in dVector:
				distanceDict[imgname] +=1
			nextNodeValue = curr_clusterer.predict([vector])[0]
			currHeightList.append(nextNodeValue)
	listOfDistances = distanceDict.items()
	sortedList = sorted(listOfDistances, key=lambda x:x[1], reverse=True)
	for i in range(10):
		print sortedList[i]


def getSiftDescriptorOfImage(img):
	global t
	gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	sift = cv2.xfeatures2d.SIFT_create()
	kp, desc = sift.detectAndCompute(gray, None)
	if(len(desc) > t):
		zipKpDesc = zip(kp, desc)
		zipKpDescSorted = sorted(zipKpDesc, key=lambda x:x[0].response, reverse=True)
		topTKpDesc = zipKpDescSorted[:t]
		topDesc = list(zip(*topTKpDesc)[1])
		return np.asarray(topDesc)
	else:
		# print "Yes"
		return desc


def fileNumFromName(fileName):
	return int(fileName.split(".")[0].replace("ukbench", ""))


def calculateDistances():
	global kMeansClusterDict
	global distanceDict
	global allDatabaseImgList

	for img in allDatabaseImgList:
		distanceDict[img] = 0

	distanceDictkeys = distanceDict.keys()

	for key in kMeansClusterDict.keys():
		dVector = kMeansClusterDict[key][2] #dVector
		q = kMeansClusterDict[key][3]
		d = 0
		for imgdistname in distanceDictkeys:
			if imgdistname in dVector:
				d = dVector[imgdistname]
				distanceDict[imgdistname] += abs(q-d)
			else:
				distanceDict[imgdistname] += q



def getMatchScore(top_4, testFileName):
	score = 0
	testTop_4Matches = []
	for imgName in top_4:
		testTop_4Matches.append(fileNumFromName(imgName))

	testFileImgNum = fileNumFromName(testFileName)
	baseNum = testFileImgNum - (testFileImgNum%4)
	trueTop_4Matches = range(baseNum, baseNum+4)

	for trueMatch in trueTop_4Matches:
		if trueMatch in testTop_4Matches:
			score +=1
	print score
	return score


def resetTreeQi_DistanceDict():
	global kMeansClusterDict
	global distanceDict
	global qiSumTotal
	for key in kMeansClusterDict.keys():
		kMeansClusterDict[key][3] = 0.0
	distanceDict = {}
	qiSumTotal = 0


def traverseVocabTree(desc):
	global kMeansClusterDict
	global distanceDict
	global qiSumTotal
	for vector in desc:
		currHeightList = [0]
		while(len(currHeightList)!=heightVocabTree+1): #While not reached the leaf node
			currNode = kMeansClusterDict[tuple(currHeightList)]
			curr_clusterer = currNode[0] # KMeans clusterer
			currNode[3] += currNode[1] # Update the last vector
			qiSumTotal += currNode[1] # Update qiSumTotal
			# print currNode[1]
			nextNodeValue = curr_clusterer.predict([vector])[0] # Predict the cluster label for current vector at curr node
			currHeightList.append(nextNodeValue)

	#Normalise qi
	for clusterer in kMeansClusterDict:
		if(qiSumTotal == 0):
			print "gadbad hai"
			break
		else:
			kMeansClusterDict[clusterer][3] /= qiSumTotal

	calculateDistances()
	listOfDistances = distanceDict.items()
	distAscending = sorted(listOfDistances, key=lambda x:x[1])
	for i in range(10):
		print distAscending[i]
	top_4 = distAscending[:4]
	return zip(*(top_4))[0] #Return Top 4 names


def scorer():
	global allDatabaseImgList
	global pathOfImageFolder

	nDocs = len(allDatabaseImgList)

	if(nDocs%4 != 0):
		print "Not multiple of 4"
		return -1

	totalScore = 0
	bestMatchScore = 0
	nrBlocks = nDocs/4
	for blockNum in range(nrBlocks):
		curr_img_index = blockNum*4 + 3 #Choose the 4th image from the 4 image block
		fileName = allDatabaseImgList[curr_img_index]
		testImgToOpenPath = pathOfImageFolder + fileName
		img = cv2.imread(testImgToOpenPath, 1)
		desc = getSiftDescriptorOfImage(img)
		top_4 = traverseVocabTree(desc)
		resetTreeQi_DistanceDict()
		matchScore = getMatchScore(top_4, fileName)
		totalScore += matchScore
		if(matchScore>1):
			bestMatchScore += 1

	# print totalScore/float(nDocs)
	print bestMatchScore/float(nrBlocks)
	return totalScore/float(nDocs)


def loadTree():
	global kMeansClusterDict
	global allDatabaseImgList
	global clustererFileName
	kMeansClusterDict = pickle.load(open(kMeansClustererDumpPath+clustererFileName, 'r'))
	rootNode = kMeansClusterDict[tuple([0])]
	allDatabaseImgList = rootNode[2].keys()
	allDatabaseImgList.sort()
	print "Tree Load Done"

def runTests():
	loadTree()
	print scorer()


if __name__ == "__main__":
	runTests()