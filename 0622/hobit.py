# Simple image serach engine
# 4stesp to build an image serach engine
#   1:define the descriptor
#   2:index the dataset
#   3:define similarity metric
#   4:search

import numpy as np
import matplotlib.pyplot as plt
import cv2
from os import listdir
from os.path import isfile, join

import pickle

################################################################################################
# +———————————————————————————————————————————————————————————————————————————————————————————+
# | Step 1: Image Descriptor : 3D Histogram
# +———————————————————————————————————————————————————————————————————————————————————————————+
################################################################################################
class RGBHistogram:
    def __init__(self, bins):
        self.bins = bins
    def describe(self, image):
        # hist = cv2.calcHist([image],[0,1,2],None,self.bins,[0,256,0,256,0,256])
        # hist = cv2.normalize(hist,hist)
        hist = cv2.calcHist(images=[image], channels=[0,1,2], mask=None,histSize=self.bins, ranges=[0,256] * 3)
        hist = cv2.normalize(hist, dst=hist.shape)

        return hist.flatten()

################################################################################################
# +———————————————————————————————————————————————————————————————————————————————————————————+
# | Step 2: Indexing
# +———————————————————————————————————————————————————————————————————————————————————————————+
################################################################################################
indexed_dataset_path = 'indexed_dataset.pkd'
index = ()
desc = RGBHistogram([8,8,8])

dataset_dir = "/Users/shinto/image-information-science/0622/images"
image_names = [f for f in listdir(dataset_dir) if isfile(join(dataset_dir,f))]
image_paths = [join(dataset_dir, name) for name in image_names]

dataset  = [cv2.imread(img_path) for img_path in image_paths]

for k, image in zip(image_names, dataset):
    # print(k,image)
    features = desc.describe(image)
    index = features

f = open(indexed_dataset_path,'wb')
f.write(pickle.dumps(index))
f.close()

################################################################################################
# +———————————————————————————————————————————————————————————————————————————————————————————+
# | Step 3: define similarity metric
# +———————————————————————————————————————————————————————————————————————————————————————————+
################################################################################################

def chi_square_distance(histA,histB,eps=1e-10):
    d = 0.5 * np.sum([((a-b)**2)/(a+b+eps) for (a,b) in zip(histA,histB)])
    return d

################################################################################################
# +———————————————————————————————————————————————————————————————————————————————————————————+
# | Step 4:the image search engine
# +———————————————————————————————————————————————————————————————————————————————————————————+
################################################################################################
class Searcher:
    def __init__(self, index):
        self.index = index

    def search(self, queryFeatures):
        result = {}
        for (k ,features) in self.index.items():
            d = chi_square_distance(features, queryFeatures)
            result[k] = d
        
        results = sorted([(v,k) for (k,v) in results.items()])

        return results

index = pickle.loads(open(indexed_dataset_path,'rb').read())
searcher = Searcher([index])
queryImage_path = "/Users/shinto/image-information-science/0622/queries/rivendell-query.png"
queryImage = cv2.imread(queryImage_path)

# cv2.imshow('Query', queryImage)
plt.imshow(cv2.cvtColor(queryImage,cv2.COLOR_BGR2RGB))
plt.title('Query')
plt.show()

queryFeatures = desc.describe(queryImage)

montageA = np.zeros((166*5,400,3), dtype='uint8')
montageB = np.zeros((166*5,400,3), dtype='uint8')

results = searcher.search(queryFeatures)

for j in range(10):
    (score, imageName) = results[j]
    path = join(dataset_dir, imageName)

    result = cv2.imread(path)
    print('`\t{}.{} :{:3f}'.format(j*1,iamgeName,score))

    if j < 5:
        montageA[j*166:(j+1)*166,:] = result
    else:
        montageB[(j-5)*166:(j-5+1)*166,:] = result

# cv2.imshow('Results 1-5',montageA)
plt.imshow(cv2.cvtColor(montageA,cv2.COLOR_BGR2RGB))
plt.title('Results 1-5')
plt.show()
# cv2.imshow('Results 6-10',montageB)
plt.imshow(cv2.cvtColor(montageB,cv2.COLOR_BGR2RGB))
plt.title('Results 1-5')
plt.show()
cv2.waitKey(0)
# cv2.destroyAllWindows()
