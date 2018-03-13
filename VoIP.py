import cv2
import urllib2
import numpy as np
import sys

# Input the URL shown by IP Webcam
url = "192.168.0.15:8080"
stream = 'http://' + url + '/video'
print('Streaming from: ' + stream)

# Open ByteStram
videoStream = urllib2.urlopen(stream)

# Collect Bytes and Process
packets = ''
while True:
    packets += videoStream.read(1024)
    a = packets.find('\xff\xd8')
    b = packets.find('\xff\xd9')
    if a!=-1 and b!=-1:
        jpg = packets[a:b+2]
        packets = packets[b+2:]

        # decode byte stream to form video
        video = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.IMREAD_COLOR)
        # display video
        cv2.imshow('WebStream',video)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
