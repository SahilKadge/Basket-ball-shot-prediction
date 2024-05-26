import cv2

image = cv2.imread("./Files/Ball.png")
image = image[0:900, :]
image = cv2.resize(image,(0,0), None, 0.6, 0.6 )
image_Copy = image.copy()
#Convert the image from bgr to hsv
imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 

# Ball detection 
lower_range = (7, 153, 29)
upper_range = (162, 255, 235)

mask = cv2.inRange(imageHSV, lower_range, upper_range)

# Find and draw the contours 
contour , hirearchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Draw contours 
cv2.drawContours(image_Copy, contour, -1, (0,255,0), 1 )
ball_detection = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Mask image", image_Copy)
# Color 0 147 135 255 113 235
cv2.imshow("image", ball_detection)
cv2.waitKey(0)