import cv2
import urllib.request as urllib2
import numpy as np
import sys

# Input the URL shown by IP Webcam
url = "35.3.71.126:8080"
tream = 'http://' + url + '/video'
print('Streaming from: ' + tream)

# Open ByteStram
stream = urllib2.urlopen(tream)

bytes = bytes()
while True:
    bytes += stream.read(1024)
    a = bytes.find(b'\xff\xd8')
    b = bytes.find(b'\xff\xd9')
    if a != -1 and b != -1:
        jpg = bytes[a:b+2]
        bytes = bytes[b+2:]
        i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        cv2.imshow('i', i)
        if cv2.waitKey(1) == 27:
            exit(0)
