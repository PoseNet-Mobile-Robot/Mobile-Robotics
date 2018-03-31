import preprocess as Pre
import numpy as np

s = "./ShopFacade/"

p = Pre.preprocess(s)
print('number of samples created: ',p.numSamples())
img, labels = p.fetch(3)
print(img.shape, labels)
print('number of samples left: ',p.samplesLeft())
img2, labels2 = p.fetch(5)
print(img2.shape)
print('number of samples left: ',p.samplesLeft())
img3, labels3 = p.fetch(5)
print(img3.shape)
print('number of samples left: ',p.samplesLeft())
p.reset()
print('number of samples left: ',p.samplesLeft())
