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
img = resize(cv2.imread(img_file),1600)
cv2.imshow("aaa", img)



# STEP 1: Finding all the staff lines through edge detection 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
cv2.imshow("canny", resize(edges,700))

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
staff_y_positions = sorted([y1 for (x1, y1, x2, y2) in staff_lines]) # will help in finding pitch

# getting rid of potential line duplicates & merging lines within 4px
unique_ys = []
for y in sorted([y1 for (_,y1,_,_) in staff_lines]):
    if not unique_ys or abs(y - unique_ys[-1]) > 4:
        unique_ys.append(y)

# grouping individual staff lines into 5-line staves
ys = unique_ys
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
cv2.imshow("Staff Removed", resize(staffless_img,600))

# STEP 2: Finding potential musical symbols 
binary = cv2.adaptiveThreshold(staffless_img, 255, 
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY_INV, 
                               35, # block size
                               11) # constant subtracted from mean
#cv2.imshow("Binary", resize(binary),600)
cv2.imwrite("binary.png", binary)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
symbols = []
staffless_color = cv2.cvtColor(staffless_img, cv2.COLOR_GRAY2BGR)

# computing average spacing from detected staves
avg_spacing = np.mean([abs(staff[i+1] - staff[i]) for staff in staves for i in range(4)])
top_of_first_staff = min(staff[0] for staff in staves)

# filtering contours 
for contour in contours:
    area = cv2.contourArea(contour)
    if area < 20 or area > 15000:
        continue 
    x,y,w,h = cv2.boundingRect(contour)
    ratio = w / h
    if ratio < 0.15 or ratio > 4:
        continue
    if h < 0.5 * avg_spacing or h > 3.5 * avg_spacing:
        continue
    if y + h < top_of_first_staff - 10:
        continue
    # passed all the filters -> actual musical symbol candidate 
    crop = staffless_img[y:y+h, x:x+w]
    cx, cy = x + w//2, y+ h//2 
    symbols.append({"bbox": (x,y,w,h), "center": (cx,cy), "image": crop})
    cv2.rectangle(staffless_color, (x,y), (x+w, y+h), (0,0,255), 2)
#cv2.imshow("Candidates", resize(staffless_color),600)

# STEP 3: Template matching
# importing templates

#clef_template = cv2.imread("./ressource/templates/clef.png", cv2.IMREAD_GRAYSCALE)
treble_clef_template = resize(cv2.imread("./ressource/templates/treble-clef.png", cv2.IMREAD_GRAYSCALE),100)
bass_clef_template = resize(cv2.imread("./ressource/templates/bass-clef.png", cv2.IMREAD_GRAYSCALE),75)
#wholespace_template = cv2.imread("./ressource/templates/whole-space.png", cv2.IMREAD_GRAYSCALE)
whole_template = resize(cv2.imread("./ressource/templates/whole-note.png", cv2.IMREAD_GRAYSCALE),15)
# half_template = cv2.imread("./ressource/templates/half.png", cv2.IMREAD_GRAYSCALE)
half_template = resize(cv2.imread("./ressource/templates/half-note-space.png", cv2.IMREAD_GRAYSCALE),15)
#quarter_template = cv2.imread("./ressource/templates/quarter.png", cv2.IMREAD_GRAYSCALE)
quarter_template = resize(cv2.imread("./ressource/templates/quarter-note-line.png", cv2.IMREAD_GRAYSCALE),15)
eighth_up_template =resize(cv2.imread("./ressource/templates/eighth-note-line.png", cv2.IMREAD_GRAYSCALE),75)
barred_eighth_template =resize(cv2.imread("./ressource/templates/barred-eighths.png", cv2.IMREAD_GRAYSCALE),75)
#eighth_down_template =resize( cv2.imread("./ressource/templates/eighth-note-space.png", cv2.IMREAD_GRAYSCALE),50)
#sharp_template = cv2.imread("./ressource/templates/sharp.png", cv2.IMREAD_GRAYSCALE)
sharp_template = resize(cv2.imread("./ressource/templates/sharp-space.png", cv2.IMREAD_GRAYSCALE),50)
flat_template = resize(cv2.imread("./ressource/templates/flat-space.png", cv2.IMREAD_GRAYSCALE), 50)
whole_rest_template = resize(cv2.imread("./ressource/templates/whole-rest.png", cv2.IMREAD_GRAYSCALE), 15)
half_rest_template = resize(cv2.imread("./ressource/templates/half-rest.png", cv2.IMREAD_GRAYSCALE), 15)
quarter_rest_template = resize(cv2.imread("./ressource/templates/quarter-rest.png", cv2.IMREAD_GRAYSCALE), 15)
eighth_rest_template = resize(cv2.imread("./ressource/templates/eighth-rest.png", cv2.IMREAD_GRAYSCALE), 15)
dot_template = resize(cv2.imread("./ressource/templates/dot.png", cv2.IMREAD_GRAYSCALE),20)
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
    rectangles, _ = cv2.groupRectangles(rectangles, groupThreshold=5, eps=0.6)
    
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
            if ys[0] - 50 < p[1] < ys[-1] + 25:
                found = True
                break
        if found:
            filtered.append(p)
    return filtered

def delete_left(rectangles,left):
    filtered = []
    for p in rectangles:
            if left  < p[0]:
                filtered.append(p)
    return filtered


# making colored version of img so rectangles show up
staffless_color = cv2.cvtColor(staffless_img, cv2.COLOR_GRAY2BGR)
found_treble_clefs= findSymbol(staffless_img,treble_clef_template, 0.3) 
found_bass_clefs= findSymbol(staffless_img,bass_clef_template, 0.35)
found_halfs= findSymbol(staffless_img,half_template, 0.45)
found_quarters= findSymbol(staffless_img,quarter_template, 0.55)
found_sharps= findSymbol(staffless_img,sharp_template, 0.55)
found_wholes= findSymbol(staffless_img,whole_template, 0.535)
found_dots= findSymbol(staffless_img,dot_template, 0.70)
found_eigths= findSymbol(staffless_img,eighth_up_template, 0.47)
#found_barred_eigths= findSymbol(staffless_img,barred_eighth_template, 0.2)


# deleting everything that's not in the staves (ie text)
found_treble_clefs=delete_out_of_staff(found_treble_clefs,staves)
found_bass_clefs=delete_out_of_staff(found_bass_clefs,staves)
found_halfs=delete_out_of_staff(found_halfs,staves)
found_quarters= delete_out_of_staff(found_quarters,staves)
found_sharps= delete_out_of_staff(found_sharps,staves)
found_wholes=delete_out_of_staff(found_wholes,staves)
found_dots= delete_out_of_staff(found_dots,staves)
found_eigths= delete_out_of_staff(found_eigths,staves)

found_quarters = delete_left(found_quarters,found_treble_clefs[0][0]+found_treble_clefs[0][2]) #temporary solution
found_halfs = delete_left(found_halfs,found_treble_clefs[0][0]+found_treble_clefs[0][2])
found_wholes= delete_left(found_wholes,found_treble_clefs[0][0]+found_treble_clefs[0][2])
found_dots=delete_left(found_dots,found_treble_clefs[0][0]+found_treble_clefs[0][2])

#found_quarters=remove_inner_boxes(found_barred_eigths,found_quarters)
#found_quarters=remove_inner_boxes(found_eigths,found_quarters)
found_quarters=remove_inner_boxes(found_wholes,found_quarters)
found_halfs=remove_inner_boxes(found_eigths,found_halfs)
found_halfs=remove_inner_boxes(found_wholes,found_halfs)
found_dots=remove_inner_boxes(found_halfs,found_dots)
found_dots=remove_inner_boxes(found_quarters,found_dots)
found_dots=remove_inner_boxes(found_wholes,found_dots)



draw_rect(found_treble_clefs,staffless_color,(0,0,0))
draw_rect(found_bass_clefs,staffless_color,(0,0,0))
draw_rect(found_halfs,staffless_color,(255,0,0))
draw_rect(found_quarters,staffless_color,(0,0,255))
draw_rect(found_quarters,staffless_color,(0,0,255))
draw_rect(found_sharps,staffless_color,(0,255,0))
draw_rect(found_wholes,staffless_color,(255,255,0))
draw_rect(found_dots,staffless_color,(0,125,255))
draw_rect(found_eigths,staffless_color,(255,0,255))
#draw_rect(found_barred_eigths,staffless_color,(255,0,255))

# We create a note object, and then we find on which staff it is
def sort_in_staff(staff_notes, staves, found, duration, type):
    for p in found:
        x,y,w,h = p
        cx = x + w // 2 # getting the centerpoint of the notehead
        cy = y + h // 2
        if type=="barred_eigth":
                note1= Note (0, duration, cx, cy+10,"eigth")
                note2= Note (0, duration, cx+10, cy,"eigth")
        note = Note (0, duration, cx, cy,type)
        for i in range (len(staves)):
            ys = sorted(staves[i])
            if ys[0] - 50 < cy < ys[-1] + 50:
                staff_notes[i].append(note)
                if type=="barred_eigth":
                    staff_notes[i].append(note1)
                    staff_notes[i].append(note2)
                break
    return staff_notes

#We create an object where all our notes we be organized by staves
staff_notes = []
for i in range(len(staves)):
    staff_notes.append([])

#We call sort_in_staff for every type of note, giving them a duration and a type name (for readability)
staff_notes=sort_in_staff(staff_notes,staves,found_treble_clefs,0,"treble_clef")
staff_notes=sort_in_staff(staff_notes,staves,found_bass_clefs,0,"bass_clef")
staff_notes=sort_in_staff(staff_notes,staves,found_sharps,0,"sharp")

staff_notes=sort_in_staff(staff_notes,staves,found_quarters,0.25,"quarter")
staff_notes=sort_in_staff(staff_notes,staves,found_halfs,0.5,"half")
staff_notes=sort_in_staff(staff_notes,staves,found_wholes,1,"whole")
staff_notes=sort_in_staff(staff_notes,staves,found_dots,0,"dot")
staff_notes=sort_in_staff(staff_notes,staves,found_eigths,0.125,"eigth")
# staff_notes=sort_in_staff(staff_notes,staves,found_barred_eigths,0.125,"barred_eigths")

#staff_notes=sort_in_staff(staff_notes,staves,found_sharps,0,"sharps") 
# i commented sharps out just bc they're not "notes" per se but idk what you want to do with those

# We loop through each staff and assign a pitch 
# treble clef: space above top line -> space below bottom line
treble_names = ["G5","F5","E5","D5","C5","B4","A4","G4","F4","E4","D4","C4"]
treble_vals = [ 79 , 77 , 76 , 74 , 72 , 71 , 69 , 67 , 65 , 64 , 62, 60 ]

bass_vals = [59 , 57 , 55 , 53 , 52 , 50 , 48 , 47, 45 , 43, 42,40]

note_start= False
for si, staff in enumerate(staff_notes):
    if staff[0].type=="bass_clef":
        pitch_vals=bass_vals
    else:
        pitch_vals=treble_vals
    positions = staff_info[si]["pitch_positions"]
    for note in staff:
        if note.type!="treble_clef" and note.type!="bass_clef" :
            #print(positions[1]-positions[0])
            cy = note.y 
            # find nearest staff line/space position
            diffs = [abs(cy - pos) for pos in positions]
            idx = diffs.index(min(diffs))
            if positions[-1]-7<cy:
                idx+=1
            if not note_start:
                match note.type:
                    case "sharp":
                        pitch_vals[idx]+=1
                        if idx<4:
                            pitch_vals[idx+7]+=1
                    case "flat":
                        pitch_vals[idx]-=1
                        if idx<4:
                            pitch_vals[idx+7]-=1
                    case _:
                        note_start = True
                        if note.type=="eigth":
                            idx+=3
                        note.pitch = pitch_vals[idx]                
            else:
                if note.type=="eigth":
                            idx+=3
                note.pitch = pitch_vals[idx]


#finally, we sort them from left to right in the array so they're in order
for staff in staff_notes:
    staff.sort(key=lambda note: note.x)

#Print them to double check everything is aok
for i in range(len(staff_notes)):
    print("Staff n°" + str(i+1) + ":")
    #for j in range(len(staff_notes[i])):
    #    print(staff_notes[i][j].type)
    for note in staff_notes[i]:
        # printing note type, pitch, and coordinates (sanity check before converting to MIDI)
        print(f"{note.type} -> {note.pitch} (x={note.x}, y={note.y})") 
    print("\n")


# fusing everything into a single array of """notes"""
treble_notes = []
bass_notes = []
for i in staff_notes:
    if i[0].type=="treble_clef":
        for j in i:
            treble_notes.append(j)
    else:
        for j in i:
            bass_notes.append(j)


MyMIDI = MIDIFile(numTracks=2)  # One track, defaults to format 1 (tempo track is created
                      # automatically)
MyMIDI.addTempo(0, 0, tempo=60)

time=0
#for i,note in enumerate(all_notes):
for i in range (len(treble_notes)):
    note=treble_notes[i]
    if note.duration!=0:
        if i!=len(treble_notes) -1 and treble_notes[i+1].type=="dot":
            note.duration*=1.5
        MyMIDI.addNote(track=0, channel=0, pitch=note.pitch, time=time, duration=note.duration, volume=100)
        time=time+note.duration

time=0
#for i,note in enumerate(all_notes):
for i in range (len(bass_notes)):
    note=bass_notes[i]
    if note.duration!=0:
        if i!=len(bass_notes) -1 and bass_notes[i+1].type=="dot":
            note.duration*=1.5
        MyMIDI.addNote(track=1, channel=0, pitch=note.pitch, time=time, duration=note.duration, volume=100)
        if i<len(bass_notes)-1:
            if note.x!=bass_notes[i+1].x:
                print(str(note.x) + " , " + str(bass_notes[i+1].x))
                time=time+note.duration
            else:
               print("homo")
        else:
                time=time+note.duration

with open("output.mid", "wb") as output_file:
    MyMIDI.writeFile(output_file)

#cv2.imshow("Matchtemplate", resize(result))
cv2.imshow("Everything (staffless)", resize(staffless_color,600))
cv2.imwrite("symbols.png", staffless_color)


# resizing

cv2.waitKey(0)
cv2.destroyAllWindows()
