# Eigenface-Facial-Recognition
This is an experiment in using eigenfaces as a preprocessing method to improve existing facial recognition software. The theory was that an eigenface photo would perform better than a normal photo when used with standard facial recognition packages.

This experiment also explores the idea that median and mode versions of the eigenface could potentially be more effective than the standard "average eigenface".

*Developed using Python 3.9*
*November 2020*
___

## Installing Packages
This project requires the use of the python face_recognition package, which is difficult to install at best.

For more information about installation, check out [this link](https://pypi.org/project/face-recognition/).

**Trying to use pip will fail if you do not follow the instructions in the link.**

___

## Running the Project + Short Explanation
Once all packages have been installed properly, simply **run main**. This will capture your face using the default camera for your computer (usually the webcam), generate the eigenfaces (mean, median, and mode), and then encode all 3 eigenfaces as well as a control photo to be used for facial recognition. Footage from the webcam will then be displayed on the screen. When a face is detected in the video, it will be compared against the control, mean, median and mode faces in an attempt to recognize it. The unknown face will be compared to the previously encoded faces (control, mean, median, or mode) and the encoding that is the best recognizer of the unknown face will be shown on the screen. This allows each encoding to be compared head-to-head in real time.

**To run the code piece by piece**, this is the recommended order:
1. face_capture.py
2. metric_test.py

___

## Important Notes
+ To use a prerecorded test video, find the instructions in the comments included in main
+ Ensure your entire head remains in frame during the face capture.
+ The control photo that is used in the tests is capture5.jpg from face_capture. It could be any photo, but this seemed most convenient.

___

## What I Learned:
+ Image manipulation using Pillow and Numpy.
+ The basic concepts behind facial recognition.
+ the Face-recognition package makes using facial recognition software incredibly easy.
