import sys
import subprocess
import cv2
import time
import numpy as np
from random import randint
from midiutil import MIDIFile

img_file = sys.argv[1:][0]
img = cv2.imread(img_file)
cv2.imshow("aaa",img)
k = cv2.waitKey(0) #Makes it so the images doesn't close instantly


