import numpy as np
from scipy import spatial
import os

vector_dimension = 32
inputPath=r"D:\Bird\GitHub\yolact\coef"


    
def hausdorffDistance(setA, setB, distance_func):
    AtoB = []
    for i in setA:
        ItoB = []
        for j in setB:
            ItoB.append(distance_func(i,j))
        AtoB.append(min(ItoB))
    
    BtoA = []
    for i in setB:
        ItoA = []
        for j in setA:
            ItoA.append(distance_func(i,j))
        BtoA.append(min(ItoA))
    
    return max(max(AtoB), max(BtoA))

for filei in os.listdir(inputPath):
    for filej in os.listdir(inputPath):
        
        objectA = os.path.splitext(filei)[0]
        objectB = os.path.splitext(filej)[0]

        fileA = open(os.path.join(inputPath, objectA)+'.txt', 'r')
        fileB = open(os.path.join(inputPath, objectB)+'.txt', 'r')

        setA = np.empty(shape=(0,vector_dimension))
        setB = np.empty(shape=(0,vector_dimension))

        rawData = fileA.readline()
        while rawData:
            rawData = rawData.split()
            if int(rawData[1]):
                setA = np.concatenate((setA, np.array([rawData[2:]]).astype(float)))
            rawData = fileA.readline()
        fileA.close()
            
        rawData = fileB.readline()
        while rawData:
            rawData = rawData.split()
            if int(rawData[1]):
                setB = np.concatenate((setB, np.array([rawData[2:]]).astype(float)))
            rawData = fileB.readline()
        fileB.close()
        
        euclidean_Distance_between_AB = hausdorffDistance(setA, setB, spatial.distance.euclidean)
        cosine_Distance_between_AB = hausdorffDistance(setA, setB, spatial.distance.cosine)
        
        print('distance between %25s & %25s : cosine[%.4f] euclidean[%.4f]'%( objectA, objectB, cosine_Distance_between_AB, euclidean_Distance_between_AB))