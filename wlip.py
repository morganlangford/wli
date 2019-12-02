# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 16:03:29 2015

@author: James
"""


import numpy as np
#import matplotlib.pyplot as plt
import cv2

cap = cv2.VideoCapture('3.h264')

xtr = 665
ytr = 421

xsb = 681
ysb = 240

trdata = np.zeros(1700)
sbdata = np.zeros(1700)
tvals = np.arange(0,1700,1)
nframes = 710
startframe = 700
count= 0
fcount = 0

while(fcount<nframes):
    ret, frame = cap.read()
    
    if not frame is None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    count+=1
    
    if count >= startframe and fcount<nframes-1 :
        if not frame is None:
            trdata[fcount] = gray[xtr,ytr]
            sbdata[fcount] = gray[xsb,ysb]
                
        fcount+=1
        
    cv2.imshow('frame',gray)
    if (cv2.waitKey(1) == 'q') :
        break

#plt.plot(tvals,trdata,'r',tvals,sbdata,'b')
#plt.show()

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.waitKey(1)

