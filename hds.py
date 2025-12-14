import sys
import cv2
import numpy as np
from midiutil import MIDIFile


# resizing function (to be used for displaying images only)
def resize(img, scale): 
    h, w = img.shape[:2]
    coeff = scale / h
    img = cv2.resize(img, (int(w * coeff), int(h * coeff)))
    return img 
class Note:
    def __init__(self,pitch,duration,x,y,type):
        self.type=type
        self.pitch = pitch
        self.duration = duration
        self.x=x
        self.y=y

# importing sheet file
img_file = sys.argv[1]
img = cv2.imread(img_file)
cv2.imshow("original", resize(img,600))




# STEP 1: Finding all the staff lines through edge detection 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
cv2.imshow("canny", resize(edges,3000))
# making a horizontal structuring element (w:50px, h:1px)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
horizontal = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
# Hough transform to detect long horizontal lines 
lines = cv2.HoughLinesP(
    horizontal, 
    rho = 1,
    theta = np.pi / 180,
    threshold = 1500,
    minLineLength = img.shape[1] * 0.5, # at least half the width of the page
    maxLineGap = 100
)

cv2.imshow("lines", resize(img,700))

staff_lines = []
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # keeping only ~almost~ horizontal lines
        if abs(y1 - y2) < 3: # small tolerance for tilt
            staff_lines.append((x1, y1, x2, y2))
            
staff_y_positions = sorted([y1 for (x1, y1, x2, y2) in staff_lines]) # will help in finding pitch

# getting rid of potential line duplicates & merging lines within 4px
unique_ys = []
for y in sorted([y1 for (_,y1,_,_) in staff_lines]):
    if not unique_ys or abs(y - unique_ys[-1]) > 10:
        unique_ys.append(y)
        cv2.line(img, (200, y), (2000, y), (0, 255, 0), 2)

# grouping individual staff lines into 5-line staves
ys = unique_ys
print(len(ys))
staves = []
current = [ys[0]]
for y in ys[1:]:
    print(y - current[-1])
    if abs(y - current[-1]) < 192: # rough spacing between staff lines
        current.append(y)
    else:
        if len(current) == 5:
            staves.append(current)
        current = [y]
if len(current) == 5:
    staves.append(current)

staff_info = []

# getting staff geometry & drawing bounding boxes around each stave
for staff_index, staff in enumerate(staves):
    staff = sorted(staff) # 5 y-values
    line_positions = staff[:] # exact y-values of the 5 lines
    spacing = np.mean([staff[i+1]-staff[i] for i in range(4)])
    space_positions = []
    for i in range(4):
        middle = (staff[i] + staff[i+1]) / 2
        space_positions.append(middle)
    extra_top_space = staff[0] - spacing
    extra_btm_space = staff[-1] + spacing
    # bounding box 
    top = staff[0] - 10
    bottom = staff[-1] + 10
    x1 = 0
    x2 = img.shape[1]
    cv2.rectangle(img, (x1, top), (x2, bottom), (0, 0, 255), 2)
    # putting all the pitch positions into one thing 
    pitch_positions = [extra_top_space, staff[0], space_positions[0], staff[1],
                       space_positions[1], staff[2], space_positions[2], staff[3], 
                       space_positions[3], staff[4], extra_btm_space]
    # storing all the geometry 
    staff_info.append({
        "lines": line_positions, # 5 line y-values
        "spaces": space_positions, # 4 space midpoints
        "spacing": spacing, # average staff spacing
        "top": top,
        "bottom": bottom,
        "extended_spaces": [
            extra_top_space,
            *space_positions,
            extra_btm_space
        ],
        "pitch_positions": pitch_positions
    })
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
#cv2.imshow("Staff Mask", resize(staff_mask))
cv2.imwrite("staffless.png", staffless)
staffless_img = cv2.imread("staffless.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Staff Removed", resize(staffless_img,1000))

#clef_template = cv2.imread("./ressource/clef.png", cv2.IMREAD_GRAYSCALE)
treble_clef_template = resize(cv2.imread("./ressource/treble-clef.png", cv2.IMREAD_GRAYSCALE),100)
bass_clef_template = resize(cv2.imread("./ressource/bass-clef.png", cv2.IMREAD_GRAYSCALE),100)
#wholespace_template = cv2.imread("./ressource/whole-space.png", cv2.IMREAD_GRAYSCALE)
whole_template = resize(cv2.imread("./ressource/whole-note.png", cv2.IMREAD_GRAYSCALE),25)
# half_template = cv2.imread("./ressource/half.png", cv2.IMREAD_GRAYSCALE)
half_template = resize(cv2.imread("./ressource/half-note-space.png", cv2.IMREAD_GRAYSCALE),25)
#quarter_template = cv2.imread("./ressource/quarter.png", cv2.IMREAD_GRAYSCALE)
quarter_template = resize(cv2.imread("./ressource/quarter-note-line.png", cv2.IMREAD_GRAYSCALE),25)
#eighth_up_template = cv2.imread("./ressource/eighth-note-line.png", cv2.IMREAD_GRAYSCALE)
#eighth_down_template = cv2.imread("./ressource/eighth-note-space.png", cv2.IMREAD_GRAYSCALE)
#sharp_template = cv2.imread("./ressource/sharp.png", cv2.IMREAD_GRAYSCALE)
sharp_template = resize(cv2.imread("./ressource/sharp-space.png", cv2.IMREAD_GRAYSCALE),50)
#flat_template = cv2.imread("./ressource/flat-space.png", cv2.IMREAD_GRAYSCALE)
#whole_rest_template = cv2.imread("./ressource/whole-rest.png", cv2.IMREAD_GRAYSCALE)
#half_rest_template = cv2.imread("./ressource/half-rest.png", cv2.IMREAD_GRAYSCALE)
#quarter_rest_template = cv2.imread("./ressource/quarter-rest.png", cv2.IMREAD_GRAYSCALE)
#eighth_rest_template = cv2.imread("./ressource/eighth-rest.png", cv2.IMREAD_GRAYSCALE)
dot_template = resize(cv2.imread("./ressource/dot.png", cv2.IMREAD_GRAYSCALE),15)
#cv2.imshow("template", template)

# Find locations above threshold
# try threshold: 
# 0.3 for clefs, 
# 0.45 for quarter & halfs seems good ! #win
# 0.55 for sharps
# whole spaces = not really working

def findSymbol(img,template,threshold):
    h, w = template.shape[:2]
    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)
   
    rectangles = []

    # grouping nearby detections to avoid drawing a lot of overlapping rectangles
    for pt in zip(*locations[::-1]):
        rectangles.append([pt[0], pt[1], w, h])
    # grouping overlapping rectangles  
    rectangles, _ = cv2.groupRectangles(rectangles, groupThreshold=10, eps=0.5)
    
    return rectangles

def remove_inner_boxes(a1, a2):
    keep = []

    for (x2, y2, w2, h2) in a2:
        x2b = x2 + w2
        y2b = y2 + h2

        inside_any = False
        for (x1, y1, w1, h1) in a1:
            x1b = x1 + w1 + 10
            y1b = y1 + h1 + 10

            # check if a2 box is fully inside a1 box
            if (x2 >= x1-10 and y2 >= y1-10 and
                x2b <= x1b and y2b <= y1b):
                inside_any = True
                break

        if not inside_any:
            keep.append([x2, y2, w2, h2])

    return np.array(keep)


def draw_rect(rectangles, img, color):
    for (x, y, w, h) in rectangles:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    for (x, y, w, h) in rectangles:
        cv2.rectangle(staffless_color, (x, y), (x + w, y + h), color, 2)

def delete_out_of_staff(rectangles, staves):
    filtered = []
    for p in rectangles:
        found = False
        for staff in staves:
            ys = sorted(staff)
            if ys[0] - 50 < p[1] < ys[-1] + 50:
                found = True
                break
        if found:
            filtered.append(p)
    return filtered

def delete_left(rectangles):
    filtered = []
    for p in rectangles:
            if 350 < p[0]:
                filtered.append(p)
    return filtered


# making colored version of img so rectangles show up
staffless_color = cv2.cvtColor(staffless_img, cv2.COLOR_GRAY2BGR)
found_clefs= findSymbol(staffless_img,treble_clef_template, 0.3) # should we do the same for the bass clef?
found_halfs= findSymbol(staffless_img,half_template, 0.35)
found_quarters= findSymbol(staffless_img,quarter_template, 0.65)
found_sharps= findSymbol(staffless_img,sharp_template, 0.55)
found_wholes= findSymbol(staffless_img,whole_template, 0.9)
found_dots= findSymbol(staffless_img,dot_template, 0.65)

# deleting everything that's not in the staves (ie text)
found_clefs=delete_out_of_staff(found_clefs,staves)
found_halfs=delete_out_of_staff(found_halfs,staves)
found_quarters= delete_out_of_staff(found_quarters,staves)
found_sharps= delete_out_of_staff(found_sharps,staves)
found_wholes=delete_out_of_staff(found_wholes,staves)
found_dots= delete_out_of_staff(found_dots,staves)

found_quarters = delete_left(found_quarters) #temporary solution
found_halfs = delete_left(found_halfs)
found_wholes= delete_left(found_wholes)
found_dots=delete_left(found_dots)

found_halfs=remove_inner_boxes(found_quarters,found_halfs)
found_dots=remove_inner_boxes(found_halfs,found_dots)
found_dots=remove_inner_boxes(found_quarters,found_dots)
found_dots=remove_inner_boxes(found_wholes,found_dots)



draw_rect(found_clefs,staffless_color,(0,0,0))
draw_rect(found_halfs,staffless_color,(255,0,0))
draw_rect(found_quarters,staffless_color,(0,0,255))
draw_rect(found_quarters,staffless_color,(0,0,255))
draw_rect(found_sharps,staffless_color,(0,255,0))
draw_rect(found_wholes,staffless_color,(255,255,0))
draw_rect(found_dots,staffless_color,(0,125,255))

cv2.imshow("Everything (staffless)", resize(staffless_color,600))
cv2.imwrite("symbols.png", staffless_color)

cv2.waitKey(0)
cv2.destroyAllWindows()