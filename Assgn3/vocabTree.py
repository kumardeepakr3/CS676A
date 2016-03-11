import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from collections import defaultdict
import pickle
import math


k = 11
heightVocabTree = 4
t = 400 # Max number of SIFT features to extract from an image

firstFileName = "Nil"
diSumTotal = 0
total_num_images = 0
pathOfImageFolder = "/home/deepak/Downloads/ukbenchSampleSet700/"
kMeansClustererDumpPath = "/home/deepak/Downloads/dump/"
kMeansClusterList = []
kMeansClusterDict = {}
completeSiftList = np.zeros(shape=(128,)) # Initialise to a dummy row
siftImgDict = defaultdict(list) # Dictionary of {siftVector:[List of Images]}
# voteDict = defaultdict(list) # Dictionary of {TreePath:[List of Images]}


# Dictionary of {siftVector:[List of Images]}
def updateSiftDictionary(desc, imgPath):
	global siftImgDict
	descList = desc.tolist()
	imgName = os.path.basename(imgPath)
	for vector in descList:
		siftImgDict[tuple(vector)].append(imgName)

def getSiftDescriptorOfImage(img):
	global t
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	sift = cv2.xfeatures2d.SIFT_create()
	kp, desc = sift.detectAndCompute(gray, None)

	#Take only top t descriptors based on response
	if(len(desc) > t):
		zipKpDesc = zip(kp, desc)
		zipKpDescSorted = sorted(zipKpDesc, key=lambda x:x[0].response, reverse=True)
		topTKpDesc = zipKpDescSorted[:t]
		topDesc = list(zip(*topTKpDesc)[1])
		return np.asarray(topDesc)
	else:
		# print "Less Num of SIFT features"
		return desc

#Returns list of path to each file in the 'pathOfFolder' directory
def getFilesInFolder(pathOfFolder):
	global firstFileName
	allFileFullPathList = []
	dirlist = os.listdir(pathOfFolder)
	dirlist.sort()
	first_time = True
	for file in dirlist:
		if first_time:
			first_time = False
			firstFileName = file.split(".")[0]
		allFileFullPathList.append(pathOfFolder+file)
	return allFileFullPathList


def siftForAllImages(directoryPath):
	global completeSiftList
	global total_num_images
	allFileFullPathList = getFilesInFolder(directoryPath)
	total_num_images = len(allFileFullPathList)
	print len(allFileFullPathList)
	i = 0
	for imgPath in allFileFullPathList:
		print i, imgPath
		img = cv2.imread(imgPath, 1)
		desc = getSiftDescriptorOfImage(img)
		updateSiftDictionary(desc, imgPath)
		completeSiftList = np.vstack((completeSiftList, desc)) #Combine the current img's sift with the previous ones
		i +=1
	completeSiftList = np.delete(completeSiftList, 0,0) # Remove the initial dummy row that was added on the top of the code
	return completeSiftList

def updateVoteDictionary(heightList, representativeVectors):
	global voteDict
	global siftImgDict
	key = "_".join(str(x) for x in heightList)
	for vector in representativeVectors:
		voteDict[key].append(siftImgDict[tuple(vector)])

# DVector returns Dictionary {imgName:di}
def createDVector(newData):
	global siftImgDict
	dDictionary = {}
	for vector in newData:
		imgList = siftImgDict[tuple(vector)]
		for img in imgList:
			if img in dDictionary.keys():
				dDictionary[img] +=1
			else:
				dDictionary[img] =1
	return dDictionary


def HeirarchicalKMeans(descVectors):
	global k
	global heightVocabTree
	global kMeansClustererDumpPath
	global kMeansClusterList
	global kMeansClusterDict
	global total_num_images
	global diSumTotal

	#The queue to perform BFS traversal of Heirarchical KMeans tree
	Queue = [([0],descVectors.tolist())]

	# heightList = [0]
	while(len(Queue) != 0): #Queue not empty
		(currHeightList, newData) = Queue.pop(0)
		print currHeightList
		est = KMeans(k)
		est.fit(newData)
		dVector = createDVector(newData)
		weightOfNode = math.log((1.0*total_num_images)/len(dVector.keys())) # Wi of the node
		print total_num_images, len(dVector.keys()), weightOfNode
		for key in dVector: # mi*wi
			dVector[key] *= weightOfNode
			diSumTotal += dVector[key]

		# ADD this KMeans CLUSTER ESTIMATOR AT CURRENT NODE to the dictionary
		kMeansClusterDict[tuple(currHeightList)] = [est, weightOfNode, dVector, 0.0]
		labels = est.labels_
		
		vectorWithLabel = zip(newData, labels)

		#Group the sift feature vectors that lie in same cluster based on label assigned to it
		groupedData = [[y[0] for y in vectorWithLabel if y[1]==x] for x in range(k)]

		for i in range(len(groupedData)):
			print i
			if(len(currHeightList) != heightVocabTree):
				currHeightListCopy = currHeightList[:]
				currHeightListCopy.append(i)
				Queue.append((currHeightListCopy, groupedData[i]))

	# Normalising di
	for key in kMeansClusterDict.keys():
		dVector = kMeansClusterDict[key][2]
		for key2 in dVector:
			dVector[key2] /= diSumTotal

	print "Saving Cluster"
	pickle.dump(kMeansClusterDict, open(kMeansClustererDumpPath+"ClustererDict_"+str(firstFileName)+"_"+str(total_num_images)+"_"+str(k)+"_"+str(heightVocabTree)+".pckl", 'wb'))


if __name__ == "__main__":
	descVectors = siftForAllImages(pathOfImageFolder)
	print "Done with extracting Descriptors of Images",descVectors.shape
	print "Now running Heirarchical KMeans"
	kList = [1]
	hList = []
	HeirarchicalKMeans(descVectors)