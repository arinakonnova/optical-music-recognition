import sys
import cv2
import numpy as np

# importing sheet file
img_file = sys.argv[1]
img = cv2.imread(img_file)
#cv2.imshow("aaa", img)

# resizing function (to be used for displaying images only)
def resize(img): 
    h, w = img.shape[:2]
    scale = 600 / h
    img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img 

# edge detection 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
# making a horizontal structuring element (w:50px, h:1px)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
horizontal = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
# Hough transform to detect long horizontal lines 
lines = cv2.HoughLinesP(
    horizontal, 
    rho = 1,
    theta = np.pi / 180,
    threshold = 100,
    minLineLength = img.shape[1] * 0.5, # at least half the width of the page
    maxLineGap = 10
)

staff_lines = []
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # keeping only ~almost~ horizontal lines
        if abs(y1 - y2) < 3: # small tolerance for tilt
            staff_lines.append((x1, y1, x2, y2))
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# grouping individual staff lines into 5-line staves
ys = sorted([y1 for (x1, y1, x2, y2) in staff_lines])
staves = []
current = [ys[0]]
for y in ys[1:]:
    if abs(y - current[-1]) < 20: # rough spacing between staff lines
        current.append(y)
    else:
        if len(current) == 5:
            staves.append(current)
        current = [y]
if len(current) == 5:
    staves.append(current)

# drawing bounding boxes around each stave
for staff in staves: 
    ys = sorted(staff)
    top = ys[0] - 10 # padding above the first line
    bottom = ys[-1] + 10 # padding below the last line
    # staff spans the full width of the image 
    x1 = 0
    x2 = img.shape[1]
    # drawing the actual box 
    cv2.rectangle(img, (x1, top), (x2, bottom), (0, 0, 255), 2)
# saving the image with the staves highlighted 
cv2.imwrite("staves_detected.png", img)

# masking staff lines for easier symbol detection
staff_mask = np.zeros_like(gray)
for x1, y1, x2, y2 in staff_lines:
    cv2.line(staff_mask, (x1, y1), (x2, y2), 255, 2) # thickness = 2px
kernel = np.ones((1,3), np.uint8) # applying erosion with a horizontal kernel 
thin_mask = cv2.erode(staff_mask, kernel, iterations = 1)
staffless = cv2.inpaint(gray, thin_mask, 2, cv2.INPAINT_TELEA)
cv2.imshow("Staff Mask", resize(staff_mask))
cv2.imwrite("staffless.png", staffless)
staffless_img = cv2.imread("staffless.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Staff Removed", resize(staffless_img))
# AT THIS POINT: staff lines are masked pretty well BUT some of the top and bottom edges
# (of half notes in spaces especially) are getting clipped which isn't ideal 
# What we could do is: do notehead detection first, make a notehead protection mask, 
# then remove staff line pixels only if they're not in notehead regions

# importing template

template = cv2.imread("./ressource/quarter.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow("template", template)

h, w = template.shape[:2]

# Template matching
result = cv2.matchTemplate(staffless_img, template, cv2.TM_CCOEFF_NORMED)

# Find locations above threshold
# try threshold: 
# 0.3 for clefs, 
# 0.45 for quarter & halfs seems good ! #win
# 0.55 for sharps
# whole spaces = not really working

t=0.45

locations = np.where(result >= t)

# print number of matches
# print(len(locations[0]))

# making colored version of staffless_img so rectangles show up
staffless_color = cv2.cvtColor(staffless_img, cv2.COLOR_GRAY2BGR)

# grouping nearby detections to avoid drawing a lot of overlapping rectangles
rectangles = []
for pt in zip(*locations[::-1]):
    rectangles.append([pt[0], pt[1], w, h])

# grouping overlapping rectangles
rectangles, _ = cv2.groupRectangles(rectangles, groupThreshold=1, eps=0.5)

# drawing rectangles
for (x, y, w, h) in rectangles:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
for (x, y, w, h) in rectangles:
    cv2.rectangle(staffless_color, (x, y), (x + w, y + h), (0, 0, 255), 2)

# saving result

cv2.imshow("Matchtemplate", resize(result))
cv2.imshow("Quarter Notes (staffless)", resize(staffless_color))

# resizing
img = resize(img)
cv2.imshow(str(t), img)

cv2.waitKey(0)
cv2.destroyAllWindows()
