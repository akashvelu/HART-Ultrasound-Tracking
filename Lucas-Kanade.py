import numpy as np
import cv2 
import time
cap = cv2.VideoCapture('test.avi')

#cap.read() returns true if next frame read properly, boolean value stored in ret
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

#LK params
lk_params = dict(winSize = (25, 25), maxLevel = 8, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Mouse Select Point function
def select_point(event, x, y, flags, params):
    global point, point_selected, old_points
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        point_selected = True 

        #want to overwrite old point to be new one so LK doesn't screw up
        old_points = np.array([[x, y]], dtype=np.float32)

point_selected = False
point = ()
old_points = np.array([[]])

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", select_point) #callback function: (i think) asynchronously (continues on on code) waits for mouse event, and when it does, calls select_point. 


while True:
    ret, frame = cap.read()
    if not ret:
        break 
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if point_selected is True: 
        cv2.circle(frame, point, 5, (0, 0, 255), 2)
        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)

        old_gray = gray_frame.copy()

        old_points = new_points

        x, y = new_points.ravel()
        cv2.circle(frame, (x, y), 7, (0, 255, 0), -1)
        
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break 
    time.sleep(0.01)


cap.release()
cv2.destroyAllWindows()






