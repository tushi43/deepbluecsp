# deepbluecsp
Deep Blue Citizen Service Problem complete solution
The project is to detect length, breadth and depth of pothole

To run the solution : use smoothing.py or dimensions.py for detecting pothole
to detect depth run: depth_pr.py

Python Libraires used: Opencv-python, Numpy and Matplotlib with python 3.6

For detection of shape: 
-> Smooth the image with median/bilater filter
-> Convert to Grayscale
-> Threshold the image (You can Otsu and many other thresholding)
-> find contours
-> draw contours

For detection of depth:
-> smooth the image
-> Apply homomorphic filter
-> thresholding the image
-> Remove the background
-> Apply MinAreaRect to detect the minimum length of rectangle
-> input Angle_Of_View(AOV) from mobile and vertical distance from ground
-> Calculate depth from depth function
