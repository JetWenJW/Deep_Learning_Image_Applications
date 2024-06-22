import cv2 as cv
import numpy as np

#zeros(width, height, colour(RGB))
blank = np.zeros((500, 500, 3), dtype = 'uint8')
cv.imshow('Blanck', blank)

# 1. Paint the Image a certain colour
blank[200:300, 300:400] = 0, 255, 0
cv.imshow('Green', blank)

# 2. Drawing a Rectangle
cv.rectangle(blank, (0, 0), (blank.shape[1]//2, blank.shape[1]//2), (0, 255, 0), thickness = -1)
cv.imshow('Rectangle', blank)

# 3. Draw Circle
cv.circle(blank, (250, 250), 40, (0, 0, 255), thickness = -1)
cv.imshow('Circle', blank)

# 4. Draw Line
cv.line(blank, (0, 0), (blank.shape[1]//2, blank.shape[0]//2), (255, 255, 255), thickness = 10)
cv.imshow('Line', blank)

# 5. Put Text
cv.putText(blank, 'Hello', (255, 255), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255, 0, 0), thickness = 2)
cv.imshow('Text', blank)
if 0xFF == ord('d'):
    cv.destroyAllWindows()


cv.waitKey(0)



