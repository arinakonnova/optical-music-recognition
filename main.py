import sys
import cv2
import numpy as np

# importing sheet file
img_file = sys.argv[1]
img = cv2.imread(img_file)
#cv2.imshow("aaa", img)

# importing template

template = cv2.imread("./ressource/quarter.png")
cv2.imshow("template", template)

h, w = template.shape[:2]

# Template matching
result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

# Find locations above threshold
# try threshold: 
# 0.3 for clefs, 
# 0.45 for quarter & halfs seems good ! #win
# 0.55 for sharps
# whole spaces = not really working

t=0.45

locations = np.where(result >= t)

# print number of matches
#print(len(locations[0]))

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

#resize
h, w = img.shape[:2]
scale = 800 / w   # target width = 1000px
img = cv2.resize(img, (int(w * scale), int(h * scale)))
cv2.imshow(str(t), img)

cv2.waitKey(0)
cv2.destroyAllWindows()
