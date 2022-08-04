#!/usr/bin/env python
import cv2
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int16
from cv_bridge import CvBridge,CvBridgeError
import numpy as np

cap =cv2.VideoCapture(2)
bridge=CvBridge()
now_number=0
debounce=0




def calculate_circle(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
			
		elif number==2:		
			img2 = img[0:a,y:b]
			area2=calculate_circle(img2)
			
		elif number==3:
			img3 = img[0:p,x:b]
			area3=calculate_circle(img3)
		elif number==4:
			img4 = img[p:q,x:b]
			area4=calculate_circle(img4)
		else :
			print ("someting wrong")
			break
		
		number = number+1
		
	if area2==0:
		if area1==1:
			return 1
		else:
			return 2
	elif area2==1:
		if area1==2:
			return 6
		elif area1==1 and area3==1:
			return 5
		else:
			return 3
	else:
		return 4


def talker():
    global debounce
    global now_number
    #init node/image 是 publisher的名字
    rospy.init_node('Braille',anonymous=False)
    #bulid a publisher
    pub_ori_img=rospy.Publisher('/Webcam',Image,queue_size=1)
    pub_Braille=rospy.Publisher('/Braille_result',Int16,queue_size=3)
    rec_contour=[]
    
    while not rospy.is_shutdown():
        ret, frame=cap.read()

        if not ret:
            break;
        
        
        gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #blurred
        kernel =np.ones((5,5),np.float32)/25
        blur_img=cv2.filter2D(src=gray,ddepth=-1,kernel=kernel)
        #canny
        canny_img = cv2.Canny(blur_img, 20, 160)
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

        if rec_contour is not None:
            for contour in rec_contour:
                    # displaying the image after drawing contours
                x,y,w,h = cv2.boundingRect(contour)

                #img after cutting
                pts1 = np.float32([[x,y],[x+w,y],[x,y+h],[x+w,y+h]])
                pts2 = np.float32([[0,0],[390,0],[0,290],[390,290]])
                M=cv2.getPerspectiveTransform(pts1,pts2)
                dst=cv2.warpPerspective(frame.copy(),M,(390,290))
                if rec_contour is not None:
                    cv2.imshow('result',dst)
                    rec_contour=[]
                    result=get_result(dst)
                    if(now_number==result):
                        debounce=debounce+1
                        if debounce>5:
                            pub_Braille.publish(result)
                            now_number=0
                    else:
                        now_number=result;

    

        webcam = bridge.cv2_to_imgmsg(frame,"bgr8")
        #發布消息
        pub_ori_img.publish(webcam)


        if cv2.waitKey(1)&0xFF == ord('q'):
            break

        if rospy.is_shutdown():
            cap.release()

if __name__ == '__main__':

    try:

        talker()
    except rospy.ROSInternalException:
        pass
