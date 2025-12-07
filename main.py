import sys
import subprocess
import cv2
import time
import numpy as np
from random import randint
from midiutil import MIDIFile
from matplotlib import pyplot as plt


# importing sheet file
img_file = sys.argv[1:][0]
img = cv2.imread(img_file)
cv2.imshow("aaa",img)

#importing template
staff_template= cv2.imread("./ressource/staff.png")
cv2.imshow("template",staff_template)
w, h,_ = staff_template.shape[::-1]


#findingMatches
print("Matching Finding staffs...")
res = cv2.matchTemplate(img,staff_template,cv2.TM_CCOEFF)
print("Done !")

#got this from an online example, basically figures out where the match is and draws a rectangle around it
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img,top_left, bottom_right, 120, 2)

cv2.imshow("res",res)
cv2.imshow("Result",img)
k = cv2.waitKey(0) #Makes it so the images doesn't close instantly
