import face_recognition
import os


# These functions are used to generate face encodings using face_recognition
# This should not be run as a script, only imported.

def gen_all_face_encodings():
    known_face_names_with_type = []
    known_face_names_gen_list = []

    known_face_names = []
    known_face_encodings = []

    for name in os.listdir(os.path.join('training photos', 'eigenfaces')):
        known_face_names_with_type.append(f'{name.capitalize()} Average')
        known_face_names_with_type.append(f'{name.capitalize()} Average')
        known_face_names_with_type.append(f'{name.capitalize()} Median')
        known_face_names_gen_list.append(f'{name}')

    for name in known_face_names_gen_list:

        try:

            # Load the average photo and learn how to recognize it.
            image = face_recognition.load_image_file(
                os.path.join('training photos', 'eigenfaces', name, f'{name}_avg_eigenface.jpg'))
            face_location = face_recognition.face_locations(image, 1, 'cnn')
            avg_face_encoding = \
                face_recognition.face_encodings(image, num_jitters=25, known_face_locations=face_location,
                                                model='large')[0]

            # Load the median photo and learn how to recognize it.
            image = face_recognition.load_image_file(
                os.path.join('training photos', 'eigenfaces', name, f'{name}_median_eigenface.jpg'))
            face_location = face_recognition.face_locations(image, 1, 'cnn')
            median_face_encoding = \
                face_recognition.face_encodings(image, num_jitters=25, known_face_locations=face_location,
                                                model='large')[0]

            # Load the mode photo and learn how to recognize it.
            image = face_recognition.load_image_file(
                os.path.join('training photos', 'eigenfaces', name, f'{name}_mode_eigenface.jpg'))
            face_location = face_recognition.face_locations(image, 1, 'cnn')
            mode_face_encoding = \
                face_recognition.face_encodings(image, num_jitters=25, known_face_locations=face_location,
                                                model='large')[0]

            known_face_encodings.append(avg_face_encoding)
            known_face_encodings.append(median_face_encoding)
            known_face_encodings.append(mode_face_encoding)

            known_face_names.append(f'{name.capitalize()} Average ')
            known_face_names.append(f'{name.capitalize()} Median ')
            known_face_names.append(f'{name.capitalize()} Mode ')

            print(f'{name.capitalize()} Face Encoding Generated!')
        except:
            print(f'Error with {name.capitalize()}\'s Face Encoding. Please redo face capture.')

    return known_face_names, known_face_encodings


def gen_single_face_encodings(name: str):
    known_face_names_with_type = []
    known_face_names_gen_list = []

    known_face_names = []
    known_face_encodings = []

    known_face_names_with_type.append(f'{name.capitalize()} Average')
    known_face_names_with_type.append(f'{name.capitalize()} Average')
    known_face_names_with_type.append(f'{name.capitalize()} Median')
    known_face_names_gen_list.append(f'{name}')

    for name in known_face_names_gen_list:

        try:

            # Load the average photo and learn how to recognize it.
            image = face_recognition.load_image_file(
                os.path.join('training photos', 'eigenfaces', name, f'{name}_avg_eigenface.jpg'))
            face_location = face_recognition.face_locations(image, 1, 'cnn')
            avg_face_encoding = \
                face_recognition.face_encodings(image, num_jitters=25, known_face_locations=face_location,
                                                model='large')[0]

            # Load the median photo and learn how to recognize it.
            image = face_recognition.load_image_file(
                os.path.join('training photos', 'eigenfaces', name, f'{name}_median_eigenface.jpg'))

            face_location = face_recognition.face_locations(image, 1, 'cnn')
            median_face_encoding = \
                face_recognition.face_encodings(image, num_jitters=25, known_face_locations=face_location,
                                                model='large')[0]

            # Load the mode photo and learn how to recognize it.
            image = face_recognition.load_image_file(
                os.path.join('training photos', 'eigenfaces', name, f'{name}_mode_eigenface.jpg'))
            face_location = face_recognition.face_locations(image, 1, 'cnn')
            mode_face_encoding = \
                face_recognition.face_encodings(image, num_jitters=25, known_face_locations=face_location,
                                                model='large')[0]

            known_face_encodings.append(avg_face_encoding)
            known_face_encodings.append(median_face_encoding)
            known_face_encodings.append(mode_face_encoding)

            known_face_names.append(f'{name.capitalize()} Average ')
            known_face_names.append(f'{name.capitalize()} Median ')
            known_face_names.append(f'{name.capitalize()} Mode ')

            print(f'{name.capitalize()} Face Encoding Generated!')
        except:
            print(f'Error with {name.capitalize()}\'s Face Encoding. Please redo face capture.')

    return known_face_names, known_face_encodings


if __name__ == '__main__':
    print('This is not a script. It should only be imported and not run on its own.')
