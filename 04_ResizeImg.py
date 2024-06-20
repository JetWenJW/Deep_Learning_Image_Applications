import cv2 as cv

img = cv.imread('Images/cat.jpg')

def rescaleFrame(frame, scale = 0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)

frame_resize = rescaleFrame(img)





cv.imshow('Cat', frame_resize)
#cv.imshow('Video Resize', frame_resize)

cv.waitKey(0)










