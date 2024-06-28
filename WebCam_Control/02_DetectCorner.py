import cv2 as cv
import numpy as np

# Read the image
img = cv.imread("./shape.jpg")

# Convert the image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Apply Gaussian Blur to the grayscale image for noise reduction
gray = cv.GaussianBlur(gray, (5, 5), 0)

# Enhance the image using Adaptive Histogram Equalization
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray = clahe.apply(gray)

# Adjusted SHI-TOMASI method parameters
maxCorners = 200  # Increase the maximum number of corners
qualityLevel = 0.01  # Decrease the quality level
minDistance = 5  # Decrease the minimum distance

# Detect corners
corners = cv.goodFeaturesToTrack(gray, maxCorners=maxCorners, qualityLevel=qualityLevel, minDistance=minDistance)
corners = np.int0(corners)

# Draw detected corners on the original image
for c in corners:
    x, y = c.ravel()
    img = cv.circle(img, center=(int(x), int(y)), radius=5, color=(255, 0, 0), thickness=-1)

# Display the image
cv.imshow("Shape", img)

# Wait for a key press
if cv.waitKey(0) & 0xFF == ord('q'):
    cv.destroyAllWindows()
