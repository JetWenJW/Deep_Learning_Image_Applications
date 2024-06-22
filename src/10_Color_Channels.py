import cv2 as cv
import numpy as np

# Read the image from the file
img = cv.imread('Images/paris.jpg')

# Display the original image in a window
cv.imshow('Paris', img)

# Create a blank image with the same height and width as the original image,
#  but with only one color channel
blank = np.zeros(img.shape[:2], dtype='uint8')

# Split the image into its Blue, Green, and Red color channels
b, g, r = cv.split(img)

# Create images that highlight each color channel separately
# Blue channel image
blue = cv.merge([b, blank, blank])
# Green channel image
green = cv.merge([blank, g, blank])
# Red channel image
red = cv.merge([blank, blank, r])

# Display the images for each color channel in separate windows
cv.imshow('Blue', blue)
cv.imshow('Green', green)
cv.imshow('Red', red)

# Print the shapes of the original image and the individual color channels
print(img.shape)  # Shape of the original image
print(b.shape)    # Shape of the Blue channel
print(g.shape)    # Shape of the Green channel
print(r.shape)    # Shape of the Red channel

# Merge the individual color channels back into a single image
merged = cv.merge([b, g, r])
# Display the merged image
cv.imshow('Merged Image', merged)

# Wait for a key event indefinitely or until 'd' key is pressed
if 0xFF == ord('d'):
    cv.destroyAllWindows()
cv.waitKey(0)
