import cv2 as cv

img = cv.imread('Images/cat.jpg')
cv.imshow('Cat', img)



# Wait for a key event indefinitely or until 'd' key is pressed
if 0xFF == ord('d'):
    cv.destroyAllWindows()
cv.waitKey(0)









