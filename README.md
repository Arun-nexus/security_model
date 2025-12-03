security-model (opencv hand tracking prototype)
This is a small prototype that uses a webcam feed to track a hand and check how close it gets to a virtual boundary drawn on the screen. When the hand gets too close, the system shows a big “DANGER DANGER” warning. Nothing fancy, just basic computer vision.
how to run:-
Make sure Python is installed (3.8+ works fine).
Install the required packages:
pip install -r requirements.txt
Run the script:
python security_model.py
Your webcam will turn on automatically.
how to use it
When the window opens, draw one or more rectangles using your mouse.
Each rectangle acts as a “virtual boundary”.
Put your hand in front of the camera.
The script tracks your hand using a simple HSV skin mask + contour detection.
Based on the distance between your fingertip and the rectangle, the system switches states:
-safe
-warning
-danger
In the danger state, a large red “danger danger” overlay shows up.
On Windows, it also plays a short beep (optional).
what’s inside
basic opencv pipeline
hsv skin segmentation
largest contour as the hand
fingertip detection using contour extrema
distance check to a user-drawn rectangle
simple state logic (safe / warning / danger)
works on cpu and targets >8 fps

why this exists

This was built as a quick assignment-style demo to show a minimal real-time interaction system using classical computer vision, without mediapipe or pose APIs. It’s not production code, just a lightweight proof-of-concept.
