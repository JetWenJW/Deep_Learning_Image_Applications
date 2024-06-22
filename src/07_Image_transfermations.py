import cv2 as cv
import numpy as np

img = cv.imread('Images/paris.jpg')
cv.imshow('Paris', img)

# Translation
def translate(img, x, y):
    transNet = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transNet, dimensions)

# -x --> left
# -y --> Up
# x --> Down
# y --> Right
translated = translate(img, 100, 100)
cv.imshow('Translated', translated)

# Rotation
def rotate(img, angle, rotPoint = None):
    (height, width) = img.shape[:2]

    if rotPoint is None:
        rotPonit = (width//2, height//2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimension = (width, height)
    return cv.warpAffine(img, rotMat, dimension)


rotated = rotate(img, -45)
cv.imshow('Rotate', rotated)

# Resize
resized = cv.resize(img, (500, 500), interpolation = cv.INTER_CUBIC)
cv.imshow('Resize', resized)

# Flipping
flip = cv.flip(img, 0)
cv.imshow('Flip', flip)

# Cropping
cropped = img[200:400, 300:400]
cv.imshow('Cropped', cropped)


if 0xFF == ord('d'):
    cv.destroyAllWindows()
cv.waitKey(0)
