import cv2
import face_recognition
import os
from PIL import Image
from time import sleep
from generate_eigenfaces import generate_all_eigenfaces


# Gives the user a visual prompt to look at the camera.
# This is used by face_capture
def countdown(seconds: int):
    print('Look at the camera!')
    while seconds > 0:
        print(seconds * '.')
        sleep(1)
        seconds -= 1


# captures a specified number of frames using the specified video.
# vid_capture_location defaults to use the default webcam.
# num_capture_samples defaults to 50 as results tend to get worse the higher the samples
def face_capture(capture_name: str, vid_capture_location='0', num_capture_samples=50, ):
    capture_name = capture_name.lower()

    if vid_capture_location == '0':
        vid_capture_location = 0

    try:
        os.mkdir(os.path.join(f'training photos', 'capture', str(capture_name)))
    except:
        pass

    i = 0
    countdown(3)

    # This while loop activates the camera and captures the user's face the specified number of times.
    # The default number of captures is 50. This offers a good balance.
    video_capture = cv2.VideoCapture(vid_capture_location)

    while True:
        if not i >= num_capture_samples:

            # Grab a single frame of video
            ret, frame = video_capture.read()

            # Resize frame of video to 1/2 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

            face_locations = face_recognition.face_locations(small_frame)

            if face_locations:

                for face_location in face_locations:
                    # Print the location of each face in this image
                    top, right, bottom, left = face_location

                    # You can access the actual face itself like this:
                    face_image = small_frame[top - 60:bottom + 25, left - 20:right + 20]

                    # This is used to view the photo that was just captured.
                    # Keep in mind that this will open a significant number of photos on your computer.
                    # If this is desired, uncomment the next two lines.

                    # pil_image = Image.fromarray(face_image)
                    # pil_image.show()

                    cv2.imwrite(os.path.join('training photos', 'capture', capture_name, f'capture{i}.jpg'), face_image)
                    i += 1

            else:
                pass
        # Once the correct number of photos has been taken, break out of while loop
        else:
            break

        cv2.waitKey(0)
        cv2.destroyAllWindows()


# When run as a script, this module takes in a name, captures the face, and then generates
# all eigenfaces for the captured face.
if __name__ == '__main__':
    name = input('Please Input Your Name\n')
    name = name.lower()
    face_capture(capture_name=name, num_capture_samples=50)
    print('Face Capture Completed.\nGenerating Eigenfaces.\n')
    generate_all_eigenfaces(name=name)
