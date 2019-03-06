import cv2
import numpy as np
cap = cv2.VideoCapture(0)

#Create old frame
_, frame = cap.read()
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



#Lucas kanade params
lk_params = dict(winSize = (25, 25), maxLevel = 8, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
currPoint = 0
print(currPoint)


# Mouse Select Point function
def select_point(event, x, y, flags, params):
    global currPoint, pointOne, pointOne_selected, pointTwo, pointTwo_selected, old_pointsOne, old_pointsTwo
    if event == cv2.EVENT_LBUTTONDOWN:
        if currPoint == 0:
            pointOne = (x, y)
            pointOne_selected = True 
            old_pointsOne = np.array([[x, y]], dtype=np.float32)
            currPoint = 1
        elif currPoint == 1:
            pointTwo = (x, y)
            pointTwo_selected = True 
            old_pointsTwo = np.array([[x, y]], dtype=np.float32)
            currPoint = 0

pointOne_selected = False
pointTwo_selected = False

pointOne = ()
pointTwo = ()
old_pointsOne = np.array([[]])
old_pointsTwo = np.array([[]])


cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", select_point) #callback function: (i think) asynchronously (continues on on code) waits for mouse event, and when it does, calls select_point. 

while True:
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if pointOne_selected is True and pointTwo_selected is True:

        cv2.circle(frame, pointOne, 5, (0, 0, 255), 2)
        cv2.circle(frame, pointTwo, 5, (0, 0, 255), 2)

        new_pointsOne, statusOne, errorOne = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_pointsOne, None, **lk_params)
        new_pointsTwo, statusTwo, errorTwo = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_pointsTwo, None, **lk_params)

        old_gray = gray_frame.copy()
        
        old_pointsOne = new_pointsOne
        old_pointsTwo = new_pointsTwo

        xOne, yOne = new_pointsOne.ravel()
        xTwo, yTwo = new_pointsTwo.ravel()

        cv2.circle(frame, (xOne, yOne), 7, (0, 255, 0), -1)
        cv2.circle(frame, (xTwo, yTwo), 7, (0, 255, 0), -1)

        #first_level = cv2.pyrDown(frame)
        #second_level = cv2.pyrDown(first_level)

        
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break 

cap.release()
cv2.destroyAllWindows()