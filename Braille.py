import cv2
import numpy as np
from matplotlib import pyplot as plt

def calculate_circle(img):
	n=[]
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	(a,b,c)=img.shape
	
	circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1,20, param1=100, param2=30, minRadius=1, maxRadius=300)
	
	if circles is None:
		return 0
		
	else :
		for circle in circles[0]:
			# 座標行列
			x = int(circle[0])
			y = int(circle[1])
			# 半徑
			r = int(circle[2])
			# 在原圖用指定顏色標記出圓的位置
			img = cv2.circle(img, (x, y), r, (0, 0, 255), 3)
			return (len(circles[0]))

def get_result(img):

	number=1
	

	for i in range(4):
		(a,b,c)=img.shape #get (列，行，3通道）
		x=int(b*0.5) #右半邊圖的初始座標
		y=int(b*0.73)
		p=int(a*0.36)
		q=int(a*0.8)
	
		if number==1:
			img1 = img[0:a,x:y] #表示方式[列：列，行：行]
			area1=calculate_circle(img1)
			print("area1=",area1)
			
		elif number==2:		
			img2 = img[0:a,y:b]
			area2=calculate_circle(img2)
			print("area2=",area2)
			
		elif number==3:
			img3 = img[0:p,x:b]
			area3=calculate_circle(img3)
			print("area3=",area3)
		elif number==4:
			img4 = img[p:q,x:b]
			area4=calculate_circle(img4)
			print("area4=",area4)
		else :
			print ("someting wrong")
			break
		
		number = number+1
		
	if area2==0:
		if area1==1:
			print("此點字為1")
			return 1
		else:
			print("此點字為2")
			return 2
	elif area2==1:
		if area1==2:
			print("此點字為6")
			return 6
		elif area1==1 and area3==1:
			print("此點字為5")
			return 5
		else:
			print("此點字為3")
			return 3
	else:
		print("此點字為4")
		return 4


video = cv2.VideoCapture(2) #抓取畫面

# reading image
while True:
	rec_contour=[]
	success,img = video.read()

	# converting image into grayscale image
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# setting threshold of gray image
	#_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

	#blurred
	kernel = np.ones((5,5),np.float32)/25
	blur_img = cv2.filter2D(img.copy(),-1,kernel)

	#canny
	canny_img = cv2.Canny(blur_img, 20, 160)

	# using a findContours() function
	contours, _ = cv2.findContours(
		canny_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	i=0
	for contour in contours:
		if i==0:
			i=1
			continue
		# cv2.approxPloyDP() function to approximate the shape
		approx = cv2.approxPolyDP(
				contour, 0.01 * cv2.arcLength(contour, True), True)
	
		#find width/height=39/29=1.34
		x,y,w,h = cv2.boundingRect(contour)
		aspect_ratio = float(w)/h
		
		#find area
		area=cv2.contourArea(contour)
		extent=float(area)/(w*h)
		
		if len(approx) == 4 and aspect_ratio<1.345 and 1.335<aspect_ratio and extent>0.9:
			rec_contour.append(contour)
			print(len(rec_contour))
			cv2.drawContours(img, [contour], 0, (0, 0, 255), 2)

	if rec_contour is not None:
		for contour in rec_contour:
	
			# using drawContours() function
			cv2.drawContours(img, [contour], 0, (0, 0, 255), 2)

			cv2.putText(img, 'Q', (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
		
			# displaying the image after drawing contours
			#cv2.imshow('shapes', img)
			x,y,w,h = cv2.boundingRect(contour)
			
			#img after cutting
			pts1 = np.float32([[x,y],[x+w,y],[x,y+h],[x+w,y+h]])
			pts2 = np.float32([[0,0],[390,0],[0,290],[390,290]])
			M=cv2.getPerspectiveTransform(pts1,pts2)
			dst=cv2.warpPerspective(img.copy(),M,(390,290))
			cv2.imshow('result',dst)
			get_result(dst)
			
			if cv2.waitKey(0)&0XFF==ord('j'):
				cv2.destroyWindow('result')
				cv2.destroyWindow('shapes')
			
	cv2.imshow('img',img)
	cv2.imshow('canny', canny_img)
	

	
	
	if cv2.waitKey(1)&0XFF==ord('q'):
		break	
		 
video.release()

cv2.destroyAllWindows()		

