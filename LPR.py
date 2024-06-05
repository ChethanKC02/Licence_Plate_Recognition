
# Importing necessary libraries for image processing

import cv2  
import imutils  
import pytesseract 

# Setting the path for the Tesseract OCR engine
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Loading the imageṇṇ
image = cv2.imread('D:\PART-B [ B.E ]\SEM 5\Image Processing\car2.jpg')
if image is None:
    print("Error: Unable to load the image.")

# Resizing the image for better processing
image = imutils.resize(image, width=500)
cv2.imshow('Original Image', image)
cv2.waitKey(0)

# Converting the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Image", gray)
cv2.waitKey(0)

# Applying bilateral filtering for noise reduction
smooth = cv2.bilateralFilter(gray, 11, 17, 17)

# Performing edge detection using the Canny algorithm
corner = cv2.Canny(gray, 170, 200)
cv2.imshow("Highlighted edges", corner)
cv2.waitKey(0)

# Finding contours in the edge-detected image
seg, new = cv2.findContours(corner.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Drawing contours on the original image for visualization
image1 = image.copy()
cv2.drawContours(image1, seg, -1, (0, 0, 255), 3)
cv2.imshow('Edge segmentation', image1)
cv2.waitKey(0)

# Sorting contours by area and selecting the largest ones
seg = sorted(seg, key=cv2.contourArea, reverse=True)[:30]
NoPlate = None

# Drawing contours representing potential license plates
image2 = image.copy()
cv2.drawContours(image2, seg, -1, (0, 255, 0), 3)
cv2.imshow("Number plate segmentation", image2)
cv2.waitKey(0)

count = 0
name = 1

# Iterating over the selected contours to find the license plate
for i in seg:
    perimeter = cv2.arcLength(i, True)
    approx = cv2.approxPolyDP(i, 0.02 * perimeter, True)

    if (len(approx == 4)):
        NoPlate = approx
        x, y, w, h = cv2.boundingRect(i)
        
        crp_image = image[y:y + h, x:x + w]

        cv2.imwrite(str(name) + '.png', crp_image)
        name += 1

        break

# Drawing the detected license plate contour on the original image
cv2.drawContours(image, [NoPlate], -1, (0, 255, 0), 3)
cv2.imshow("Final Image", image)
cv2.waitKey(0)

# Displaying the extracted license plate image
crp_img = '1.png'
cv2.imshow('Number Plate', cv2.imread(crp_img))
cv2.waitKey(0)
