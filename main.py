import sys
import cv2
import numpy as np

# importing sheet file
img_file = sys.argv[1]
img = cv2.imread(img_file)
cv2.imshow("aaa", img)

# importing template
staff_template = cv2.imread("./ressource/clef.png")
cv2.imshow("template", staff_template)

# # get template size
# h, w = staff_template.shape[:2]

# # finding matches (correct function)
# print("Matching Finding staffs...")
# res = cv2.matchTemplate(img, staff_template, cv2.TM_CCOEFF_NORMED)
# print("Done!")

# # location of best match
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# # draw rectangle
# top_left = max_loc
# bottom_right = (top_left[0] + w, top_left[1] + h)
# cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)


# Convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(staff_template, cv2.COLOR_BGR2GRAY)

h, w = template_gray.shape[:2]

# Template matching
result = cv2.matchTemplate(img, staff_template, cv2.TM_CCOEFF_NORMED)

# Find locations above threshold
# try threshold: 0.3 forr clefs, 0.45 for quarters seems good ! #win
locations = np.where(result >= 0.3)

# print number of matches
print(len(locations[0]))

# To avoid drawing many overlapping rectangles, group nearby detections
rectangles = []
for pt in zip(*locations[::-1]):
    rectangles.append([pt[0], pt[1], w, h])

# Group overlapping rectangles
rectangles, _ = cv2.groupRectangles(rectangles, groupThreshold=1, eps=0.5)

# Draw rectangles
for (x, y, w, h) in rectangles:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Save result

cv2.imshow("Matchtemplate", result)
cv2.imshow("Staffs", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
