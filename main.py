import numpy as np
import dlib
from pkg_resources import resource_filename
import cv2
from threading import Thread
from deepface import DeepFace
import pandas as pd

class FaceRecognition:
    @staticmethod
    def trim_bounds(bbox, image_shape):
        return max(bbox[0], 0), max(bbox[1], 0), min(bbox[2], image_shape[1]), min(bbox[3], image_shape[0])

    @staticmethod
    def face_locations(image, upsample=1):
        face_detector = dlib.get_frontal_face_detector()  # use HOG
        _ret = []
        for face in face_detector(image, upsample):
            _ret.append(FaceRecognition.trim_bounds((face.left(), face.top(), face.right(), face.bottom()), image.shape))

        return _ret

    @staticmethod
    def load_image(file, pixeltype=cv2.IMREAD_COLOR):
        _image = cv2.imread(file, pixeltype)
        if _image is None:
            raise ValueError(f"Unable to load image from path: {file}")
        return np.array(_image)


    @staticmethod
    def face_encodings(image_path):
        db_path = "data/"
        results = DeepFace.find(img_path=image_path, db_path=db_path, enforce_detection=False, model_name="Facenet512")
        return results


    @staticmethod
    def encoding_distance(known_encodings, encoding_check):
        if len(known_encodings) == 0:
            return np.empty(0)

        return np.linalg.norm(known_encodings - encoding_check,     axis=1)

    @staticmethod
    def compare_encodings(known_encodings, encoding_check, tolerance=0.5):
        return list(FaceRecognition.encoding_distance(known_encodings, encoding_check) <= tolerance)

class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()

        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return

            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        frame = cv2.flip(self.frame, 1)
        return frame

    def stop(self):
        self.stopped = True

class FaceRecognitionProcess:
    def __init__(self, fx=0.0, fy=0.0, capture=None, known_encodings=[], known_names=[]):
        self.capture = capture
        self.stopped = False
        self.face_locations = []
        self.face_names = []
        self.known_encodings = known_encodings
        self.known_names = known_names
        self.fx = fx
        self.fy = fy

    def start(self):
        Thread(target=self.process, args=()).start()
        return self

    def process(self):
        while not self.stopped:

            # Grab a single frame from the live video stream
            _frame = self.capture.read()

            if not (self.fx == 0.0 and self.fy == 0.0):
                # Resize the frame of video to 1/x size for faster face recognition processing
                _temp_frame = cv2.resize(_frame, (0, 0), fx=self.fx, fy=self.fy)

                # Convert the image from BGR color (opencv uses) to RGB color
                _frame = _temp_frame[:, :, ::-1]
            else:
                # Convert the image from BGR color (opencv uses) to RGB color
                _frame = _frame[:, :, ::-1]

            # Find all the faces in the current frame of video
            _face_locations = FaceRecognition.face_locations(_frame)
            _face_names = []

            for (left, top, right, bottom) in _face_locations:
                face_image = _frame[top:bottom, left:right]
                temp_path = "temp_face.jpg"
                cv2.imwrite(temp_path, face_image)  # Save the detected face as a temporary image

                # Use DeepFace to identify the face
                results = FaceRecognition.face_encodings(temp_path)
                if results:
                    # Assuming the name is the folder name in the data directory
                    print(results)
                    df_results = results[0]
                    if not df_results.empty:
                        identity = df_results.iloc[0]['identity']
                    else:
                        identity = "Unknown"
                    if '/' in identity:
                        parts = identity.split("/")
                        name = parts[-2] if len(parts) > 1 else "Unknown"
                    else:
                        name = "Unknown"
                else:
                    name = "Unknown"

                _face_names.append(name)

            self.face_locations = _face_locations
            self.face_names = _face_names


    def stop(self):
        self.stopped = True


# Main function

def main():
    scale_factor = 0.5
    r_scale_factor = int(1 / (scale_factor if scale_factor != 0.0 else 1))

    video_capture = WebcamVideoStream(src=0).start()
    video_process = FaceRecognitionProcess(capture=video_capture, fx=scale_factor, fy=scale_factor).start()

    while True:

        if video_capture.stopped:
            video_capture.stop()
            break

        frame = video_capture.read()

        # Display the results
        locations = video_process.face_locations
        names = video_process.face_names

        for (left, top, right, bottom), name in zip(locations, names):
            # Scale up to the original size
            top *= r_scale_factor
            right *= r_scale_factor
            bottom *= r_scale_factor
            left *= r_scale_factor

            # Draw a box around the detected face  - BGR
            cv2.rectangle(frame, (left, top), (right, bottom), (244, 134, 66), 3)

            # Draw a label with a name
            cv2.rectangle(frame, (left-2, top - 35), (right+2, top), (244, 134, 66), cv2.FILLED)

            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.stop()
    video_process.stop()

    cv2.destroyAllWindows()


try:
    main()
except Exception as e:
    print(f"An error occurred: {e}")