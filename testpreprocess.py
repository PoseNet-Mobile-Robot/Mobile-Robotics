import preprocess as Pre
import numpy as np

s = "/Projects/Test/568/ShopFacade/"

p = Pre.preprocess(s)
print('number of samples created: ',p.numSamples())
print((p.fetch(60)).shape)
print('number of samples left: ',p.samplesLeft())
print((p.fetch(60)).shape)
print('number of samples left: ',p.samplesLeft())
print((p.fetch(60)).shape)
print('number of samples left: ',p.samplesLeft())
