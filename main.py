import sys
import cv2
import numpy as np

# importing sheet file
img_file = sys.argv[1]
img = cv2.imread(img_file)
cv2.imshow("aaa", img)

# importing template
staff_template = cv2.imread("./ressource/staff.png")
cv2.imshow("template", staff_template)

# get template size
h, w = staff_template.shape[:2]

# finding matches (correct function)
print("Matching Finding staffs...")
res = cv2.matchTemplate(img, staff_template, cv2.TM_CCOEFF_NORMED)
print("Done!")

# location of best match
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# draw rectangle
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

cv2.imshow("res", res)
cv2.imshow("Result", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
