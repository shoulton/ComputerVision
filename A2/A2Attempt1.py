import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys

cap = cv.VideoCapture(0)

# Exit if video not opened.
if not cap.isOpened():
    print("Could not open video")
    sys.exit()

# Read frames.
while True:
    ok, frame = cap.read()
    if not ok:
        print('Cannot read video')
        sys.exit()
    cv.imshow("preview", frame)
    # press space to capture template
    key = cv.waitKey(1) & 0xff
    if key == 32:
        template = cv.selectROI(frame, False)
        cv.imwrite("template.jpg", frame[int(template[1]):int(template[1]+template[3]), int(template[0]):int(template[0]+template[2])])
        cap.release()
        cv.destroyAllWindows()
        break

# w, h = template.shape[::-1]
# All the 6 methods for comparison in a list
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
           'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

cap2 = cv.VideoCapture(0)

while True:
    ok, frame = cap2.read()
    if not ok:
        print('Cannot read video')
        sys.exit()

    cv.imwrite("source.jpg", frame)
    method = eval(methods[5])

    img = cv.imread("source.jpg", 0)
    template = cv.imread("template.jpg", 0)
    w, h = template.shape[::-1]
    # Apply template Matching
    res = cv.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    threshold = 0.8
    loc = np.where(res >= threshold)
    bottom_right = (top_left[0] + w, top_left[1] + h)
    # should make rectangle around template in image
    if loc:
        cv.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)
        cv.putText(frame, "Placeholder", (100, 80), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
    cv.imshow("preview", frame)
    # press esc to end program
    key = cv.waitKey(1) or 0xff
    if key == 27:
        cap2.release()
        break
