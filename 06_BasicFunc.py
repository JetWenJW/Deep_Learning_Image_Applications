import cv2 as cv

img = cv.imread('Images/paris.jpg')
cv.imshow('paris', img)   #cv.imshow(name, params)

# Converting an Image to Gray scale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Blur
blur = cv.GaussianBlur(img, (5, 5), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

# Edge Cascade
canny = cv.Canny(img, 125, 175)
cv.imshow('Canny Edge', canny)

# Dilating Image
dilated = cv.dilate(canny, (3, 3), iterations = 10)
cv.imshow('Dliated', dilated)

# Eroding
eroded = cv.erode(dilated, (3, 3), iterations = 1)
cv.imshow('Eroded', eroded)

# Resize
resize = cv.resize(img, (500, 500), interpolation = cv.INTER_CUBIC)
cv.imshow('Resize', resize)

# Cropped
cropped = img[50:200, 200:400]
cv.imshow('Cropped', cropped)

if 0xFF == ord('d'):
    cv.destroyAllWindows()
cv.waitKey(0)



