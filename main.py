import os
from face_capture import face_capture
from metric_test import do_metric_test_webcam, do_metric_test
from generate_eigenfaces import generate_all_eigenfaces

if __name__ == '__main__':
    name = input('Enter Your Name:\n')

    try:

        # Ensure that your head remains fully in the frame of the camera.
        # If the head is not fully in frame, an error will be thrown.
        # If this happens, retry and ensure the head remains in the frame.
        face_capture(name)

    except:
        print('There was an error with the face capture, Please try again.')
        quit()

    generate_all_eigenfaces(name)

    print('Press the Q key to quit.')

    # The default metric test runs using the webcam.
    do_metric_test_webcam(name=name)

    # To run a metric test using a test video use the following steps
    # 1.) Put the video into the 'test videos' directory
    # 2.) Paste the name of the video into the line below
    # 3.) Comment out the do_metric_test_webcam() line above.
    # 4.) Un-comment the do_metric_test() line below.

    # do_metric_test(test_vid_path= os.path.join('test videos', '***NAME OF YOUR TEST VIDEO***'),name=name)
