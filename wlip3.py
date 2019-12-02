# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 12:02:07 2018

@author: James Hoyland
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


#X-Y postions of two pixels of interest

xtr = 665
ytr = 421

xsb = 681
ysb = 240

#Arrays for storing pixel values

trdata = np.zeros(1700)
sbdata = np.zeros(1700)

#"Time values" In reality just frame-numbers measured from start of time.
#The video plays throughthe whole file but we're only saving pixel data for a small range

tvals = np.arange(0,1700,1)

#Number of frames to measure pixel data for
nframes = 1700

#Actual movie frame to start measuring from
startframe = 500

#Counters and stuff
count= 0
fcount = 0
pauseNext = False

#Input video file name

filename = '3.h264'

#Start video capture session
cap = cv.VideoCapture(filename)
titletext = "File: " + filename
font = cv.FONT_HERSHEY_DUPLEX

while(cap.isOpened()):
    ret, frame = cap.read() #Read the next frame
    if frame is not None: #Check we successfully read the frame before doing anything with it
        count+=1
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)   #Convert to grayscale

        if count == startframe:
            basegray = gray 

        #Put captions on the image and display
        
        fntext = "Frame: {}".format(count)
        cv.putText(gray,titletext,(0,65), font, 1, (200,255,155), 2, cv.LINE_AA)
        cv.putText(gray,fntext,(0,130), font, 1, (200,255,155), 2, cv.LINE_AA)
        
        #If we're up to the frame we need, start recording the pixel values
        
        if count >= startframe and fcount<nframes:
            trdata[fcount] = gray[ytr,xtr]
            sbdata[fcount] = gray[ysb,xsb]
                
            gray = cv.subtract(gray,basegray)

            fcount+=1


        cv.imshow('frame',gray)
        
        #If we're paused wait here until button pressed (q quits)
        
        if pauseNext:
            wk = cv.waitKey(0)
            if wk & 0xFF == ord('q'):
                break
            else:
                pauseNext = False
        
    #Pause briefly and wait for button press (q quits, p pauses) changing this delay changes effective playback speed
        
    wk = cv.waitKey(3)    
        
    if wk & 0xFF == ord('q'):
        break
    elif wk & 0xFF == ord('p'):
        pauseNext = True
        
        
#Cleanup OpenCV things
cap.release()
cv.destroyAllWindows()

#Make a plot of our two pixel values

plt.plot(tvals,trdata,'r',tvals,sbdata,'b')
plt.show()

#Prepare data for saving

dataexp = np.column_stack([tvals,trdata,sbdata])

#Save data

np.savetxt(filename+'.points.csv',dataexp,'%d',', ')

#Essential wait otherwise windows may not be properly destroyed
cv.waitKey(1)
cv.waitKey(1)
cv.waitKey(1)
cv.waitKey(1)