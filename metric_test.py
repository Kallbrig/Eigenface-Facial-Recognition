import cv2
import numpy as np
import os

from face_encodings import *


# to use this function, a test video must be recorded of the person.
# the person 'name' must be in the video at 'test_vid_path'
def do_metric_test(test_vid_path: str, name: str):
    vid_len = 0

    # Get a reference to webcam 0
    video_capture = cv2.VideoCapture(test_vid_path)

    known_face_names, known_face_encodings = gen_single_face_encodings(name)

    # Load the control photo and learn how to recognize it.

    image = face_recognition.load_image_file(
        os.path.join('training photos', 'capture', name, 'capture5.jpg'))
    face_location = face_recognition.face_locations(image, 1, 'cnn')
    face_encoding = \
        face_recognition.face_encodings(image, num_jitters=25, known_face_locations=face_location,
                                        model='large')[0]

    known_face_encodings.append(face_encoding)
    known_face_names.append('Control')

    avg_distances = []
    for i in range(len(known_face_names)):
        avg_distances.append(0)

    # This is used to measure which eigenface has the highest recognition rate
    captures = {}
    for name in known_face_names:
        captures[name] = 0

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True:
        try:
            # Grab a single frame of video
            ret, frame = video_capture.read()

            # Resize frame of video to 1/2 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
            if process_this_frame:

                # Measures the total number of frames processed
                vid_len += 1

                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    print()
                    for i in range(len(known_face_names)):
                        print(f'{known_face_names[i]}: {round(face_distances[i], 3)}')
                        avg_distances[i] += face_distances[i]

                    if matches[best_match_index]:
                        captures[known_face_names[best_match_index]] += 1
                        name = known_face_names[best_match_index]

                    face_names.append(name)

            process_this_frame = not process_this_frame

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/2 size
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2

                # Draw a box around the face
                cv2.rectangle(frame, (left - 20, top - 60), (right + 20, bottom + 25), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left - 20, top - 85), (right + 20, top - 60), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 5, top - 70), font, .8, (255, 255, 255), 1)

            # Display the resulting image
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break




        except:
            # Release handle to the webcam
            video_capture.release()
            cv2.destroyAllWindows()
            break

    # Prints the statistics.
    print('\n')

    recognized_frames = 0
    for i in captures:
        recognized_frames += captures[i]

        print(f'Percentage using {i}: {round((captures[i] / vid_len) * 100, 2)}% or {captures[i]}/{vid_len}')
    print(f'Percentage total: {round((recognized_frames / vid_len) * 100, 2)}%\n')

    # This gives average distance from each encoded face
    # Lower is better.
    j = 0
    for i in captures:
        print(f'Average Distance from {i}: {round(avg_distances[j] / recognized_frames, 3)}')

        j += 1


# This function does the same as do_metric_test without the need for a test video.
# It will use the webcam and output the same results that do_metric_test would.
def do_metric_test_webcam(name: str, test_vid_path=0, ):
    vid_len = 0

    # Get a reference to webcam 0
    video_capture = cv2.VideoCapture(test_vid_path)

    known_face_names, known_face_encodings = gen_single_face_encodings(name)

    # Load the control photo and learn how to recognize it.
    image = face_recognition.load_image_file(
        os.path.join('training photos', 'capture', name, 'capture5.jpg'))
    face_location = face_recognition.face_locations(image, 1, 'cnn')
    face_encoding = \
        face_recognition.face_encodings(image, num_jitters=1, known_face_locations=face_location,
                                        model='large')[0]

    known_face_encodings.append(face_encoding)
    known_face_names.clear()
    known_face_names.append('Average')
    known_face_names.append('Median')
    known_face_names.append('Mode')
    known_face_names.append('Control')

    avg_distances = []
    for i in range(len(known_face_names)):
        avg_distances.append(0)

    # This is used to measure which eigenface has the highest recognition rate
    captures = {}
    for name in known_face_names:
        captures[name] = 0

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True:
        try:
            # Grab a single frame of video
            ret, frame = video_capture.read()

            # Resize frame of video to 1/2 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
            if process_this_frame:

                # Measures the total number of frames processed
                vid_len += 1

                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    print()
                    for i in range(len(known_face_names)):
                        print(f'{known_face_names[i]}: {round(face_distances[i], 3)}')
                        avg_distances[i] += face_distances[i]

                    if matches[best_match_index]:
                        captures[known_face_names[best_match_index]] += 1
                        name = known_face_names[best_match_index]

                    face_names.append(name)

            process_this_frame = not process_this_frame

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/2 size
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2

                # Draw a box around the face
                cv2.rectangle(frame, (left - 20, top - 60), (right + 20, bottom + 25), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left - 20, top - 85), (right + 20, top - 60), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 5, top - 70), font, .8, (255, 255, 255), 1)

            # Display the resulting image
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break




        except:
            # Release handle to the webcam
            video_capture.release()
            cv2.destroyAllWindows()
            break

    # Prints the statistics.
    print('\n')

    recognized_frames = 0
    for i in captures:
        recognized_frames += captures[i]

        print(f'Percentage using {i}: {round((captures[i] / vid_len) * 100, 2)}% or {captures[i]}/{vid_len}')
    print(f'Percentage total: {round((recognized_frames / vid_len) * 100, 2)}%\n')

    # This gives average distance from each encoded face
    # Lower is better.
    j = 0
    for i in captures:
        print(f'Average Distance from {i}: {round(avg_distances[j] / recognized_frames, 3)}')

        j += 1


# When run as a script, this module takes in a name, and then runs a metric test using the webcam

if __name__ == '__main__':
    name = input('Input the name of the person to be tested.\n')
    name = name.lower()
    do_metric_test_webcam(name)
