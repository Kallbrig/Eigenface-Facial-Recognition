from face_capture import face_capture
from metric_test import do_metric_test_webcam
from generate_eigenfaces import generate_all_eigenfaces

if __name__ == '__main__':
    name = input('Enter Your Name:\n')
    face_capture(name)
    generate_all_eigenfaces(name)
    do_metric_test_webcam(name=name)
