import cv2 as cv
import numpy as np

img = cv.imread("../Images/cat.jpg")

# Filter 2D
blur_filter = np.array([[1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1]
                 ])
blur_filter = blur_filter / 9
blur_filter_img = cv.filter2D(img, ddepth = -1, kernel = blur_filter)

# No_Blur
no_filter = np.array([[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]
                   ])
no_filter = no_filter / 9
No_Filter_img = cv.filter2D(img, ddepth = -1, kernel = no_filter)

# BLUR
blur_img = cv.blur(img, ksize = (111, 111))

# GAUSSIAN BLUR
Gaussian_img = cv.GaussianBlur(img, ksize = (11, 11), sigmaX = 3, sigmaY = 30)

# SHARPEN
sharpen_filter = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]
                            ])
shapr_img = cv.filter2D(img, ddepth = -1, kernel = sharpen_filter)

# EDGE DETECTION
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray_img = cv.GaussianBlur(img, ksize = (3, 3), sigmaX = 1, sigmaY = 1)
edges = cv.Laplacian(gray_img, ddepth = -1)

cv.imshow("Cat ~", edges)
cv.waitKey(0)
cv.destroyAllWindows()
