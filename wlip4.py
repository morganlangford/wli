

# -*- coding: utf-8 -*-

"""

Created on Wed Nov 21 12:02:07 2018

 

@author: James Hoyland & Morgan Langford

"""

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import numpy as np

import cv2 as cv

from matplotlib import cm

from matplotlib.ticker import LinearLocator, FormatStrFormatter

import numpy as np

 

# Define number of rows, columns and frames in the bitmap images

 

xrow = 960

ycolumn = 1280

frame = 1

 

# Create arrays for storing the first image so we can subtract it from all other images

 

firstframe = np.zeros((xrow,ycolumn))

 

#Array for storing max values

 


"""

#"Time values" In reality just frame-numbers measured from start of time.

 

tvals = np.arange(0,nframes,1)

"""

#Counters and variables

x = 0 # x position

y = 0 # y position

count= 0

fcount = 0 # frame count

 

 

#Input video file name

 

filename = '3.h264'

 

#Start video capture session

cap = cv.VideoCapture(filename)

 

while cap.isOpened():

    ret, frame = cap.read() #Read the next frame

    if frame is not None: #Check we successfully read the frame before doing anything with it

        count+=1

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #Convert to grayscale
        
        print(count)
        
        if count ==100:
            cap.release()

        if count == 1:

            

            xrow, ycolumn = gray.shape
            maxdata = np.zeros((xrow,ycolumn))

            heights = np.zeros((xrow,ycolumn))

            while x < xrow: # go through the rows

                while y < ycolumn: # go through the column

                    firstframe[x][y] = gray[x,y]

                    y+=1

                x+=1

        # Reset counters

        x = 0

        y = 0

        # Now go through the frames looking for max gray value

        while x < xrow:
            y=0

            while y < ycolumn:

                pixelval = gray[x,y] - firstframe[x][y]

                if abs(pixelval) > maxdata[x][y]:

                    maxdata[x][y] = abs(pixelval)

                    heights[x][y] = count

                y+=1

            x+=1

        fcount+=1

#Cleanup OpenCV things

cap.release()

cv.destroyAllWindows()

 

#Make a 3D plot the max data including what frame they hit their max

 

fig = plt.figure()

ax = fig.gca(projection='3d')

nx, ny = xrow, ycolumn
x = range(nx)
y = range(ny)

X, Y = np.meshgrid(y, x)

# Plot the surface.

print("x: ", X.shape)
print("x: ", Y.shape)
print("x: ", heights.shape)

surf = ax.plot_surface(X,Y,heights, cmap=cm.viridis, linewidth=0, antialiased=False)

#surf = ax.imshow(heights)

# Customize the z axis.

#ax.set_zlim(-1.01, 1.01)

#ax.zaxis.set_major_locator(LinearLocator(10))

#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

 

# Add a color bar which maps values to colors.

#fig.colorbar(surf, shrink=0.5, aspect=5)

 

plt.show()

 

#Prepare data for saving

 

#dataexp = np.column_stack([tvals,trdata,sbdata])

 

#Save data

 

#np.savetxt(filename+'.points.csv',dataexp,'%d',', ')

 

#Essential wait otherwise windows may not be properly destroyed

cv.waitKey(1)

cv.waitKey(1)

cv.waitKey(1)

cv.waitKey(1)

