import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys


def createTemplate():
    cap = cv.VideoCapture(0)

    # Exit if video not opened.
    if not cap.isOpened():
        print ("Could not open video")
        sys.exit()

    # Read frames.
    while True:
        ok, frame = cap.read()
        if not ok:
            print ('Cannot read video')
            sys.exit()
        cv.imshow("create template", frame)
        # press space to capture template
        key = cv.waitKey(1) & 0xff
        if key == 32:
            template = cv.selectROI(frame, False)
            label = input("Name your object:")
            cv.imwrite("template.jpg", frame[int(template[1]):int(template[1] + template[3]),
                                       int(template[0]):int(template[0] + template[2])])
            template = cv.imread("template.jpg", 0)
            cap.release()
            cv.destroyAllWindows()
            break
    return (template, label)


# w, h = template.shape[::-1]

def recognize(temp1, temp2, temp3):
    # All the 6 methods for comparison in a list
    methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
               'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

    cap2 = cv.VideoCapture(0)

    count = 1
    while True:
        ok, frame = cap2.read()
        if not ok:
            print ('Cannot read video')
            sys.exit()
        if count == 0:

            count = 1

            cv.imwrite("source.jpg", frame)
            method = eval(methods[5])

            img = cv.imread("source.jpg", 0)
            # template = cv.imread("template.jpg",0)
            w1, h1 = temp1[0].shape[::-1]
            w2, h2 = temp2[0].shape[::-1]
            w3, h3 = temp3[0].shape[::-1]
            # Apply template Matching
            res1 = cv.matchTemplate(img, temp1[0], method)
            res2 = cv.matchTemplate(img, temp2[0], method)
            res3 = cv.matchTemplate(img, temp3[0], method)
            # LEFT OVER CODE PROBABLY WON"T NEED
            min_val1, max_val1, min_loc1, max_loc1 = cv.minMaxLoc(res1)
            min_val2, max_val2, min_loc2, max_loc2 = cv.minMaxLoc(res2)
            min_val3, max_val3, min_loc3, max_loc3 = cv.minMaxLoc(res3)
            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                top_left1 = min_loc1
                top_left2 = min_loc2
                top_left3 = min_loc3
            else:
                top_left1 = max_loc1
                top_left2 = max_loc2
                top_left3 = max_loc3
            threshold = 0.1
            loc1 = np.where(res1 <= threshold)
            loc2 = np.where(res2 <= threshold)
            loc3 = np.where(res3 <= threshold)
            for pt in zip(*loc1[::-1]):
                cv.rectangle(frame, pt, (pt[0] + w1, pt[1] + h1), (0, 0, 255), 2)
                cv.putText(frame, temp1[1], (50, 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            for pt in zip(*loc2[::-1]):
                cv.rectangle(frame, pt, (pt[0] + w2, pt[1] + h2), (255, 0, 0), 2)
                cv.putText(frame, temp2[1], (150, 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
            for pt in zip(*loc3[::-1]):
                cv.rectangle(frame, pt, (pt[0] + w3, pt[1] + h3), (0, 255, 0), 2)
                cv.putText(frame, temp3[1], (250, 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            # LEFT OVER CODE PROBABLY WON"T NEED
            bottom_right1 = (top_left1[0] + w1, top_left1[1] + h1)
            bottom_right2 = (top_left2[0] + w2, top_left2[1] + h2)
            bottom_right3 = (top_left3[0] + w3, top_left3[1] + h3)

            # cv.rectangle(frame,top_left1, bottom_right1, (255,0,0), 2)
            # cv.putText(frame,temp1[1],(100, 80), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)

            # cv.rectangle(frame,top_left2, bottom_right2, (0,255,0), 2)
            # cv.putText(frame,temp2[1],(100, 100), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)

            # cv.rectangle(frame,top_left3, bottom_right3, (0,0,255), 2)
            # cv.putText(frame,temp3[1],(100, 120), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
            # should make rectangle around template in image
            # if loc:
            #	cv.rectangle(frame,top_left, bottom_right, (255,0,0), 2)
            #	cv.putText(frame,"Placeholder",(100, 80), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
            cv.imshow("recognizer", frame)
            # press esc to end program
            key = cv.waitKey(1) or 0xff
            if key == 27:
                cap2.release()
                break

        else:
            count = count - 1
    print("Recognition session complete")


if __name__ == '__main__':
    # temp# is a tuple (jpg,label_string) 
    temp1 = createTemplate()
    temp2 = createTemplate()
    temp3 = createTemplate()

    recognize(temp1, temp2, temp3)



