# Optical Music Recognition Project
### Arina Konnova and Amélien Le Meur
#### Image Analysis and Computer Vision
###### December 2025

This project implements an Optical Music Recognition system that 
converts one-page sheet music in PNG format into a MIDI format that 
can then be played through a Digital Audio Workstation (DAW). It 
works for monophonous scores (those in which there is only one voice
playing one note at a time) as well as those with simple homophony
(multiple voices playing at the same time but with the same rhythmic
pattern) and polyphony (multiple voices playing at the same time, 
with not necessarily the same rhythmic pattern). This system does
not rely on any neural networks or other machine learning techniques,
using only traditional computer vision methods. The system primarily
works through template matching using OpenCV's matchTemplate function. 

Our system uses the following modules and libraries: sys, cv2, numpy, 
and midiutil. These should be installed before running the project.
To run a piece of sheet music through our system, the terminal command
from the optical-music-recognition directory is: 
% python file/path/to/omr/directory/main.py ressource/sheet/file-name.png 