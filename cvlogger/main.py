import logging as log
from time import perf_counter, time
import cv2
import numpy as np
import datetime
import mysql.connector
from openvino.runtime import Core
from api.landmarks_detector import LandmarksDetector
from api.face_detector import FaceDetector
from api.face_identifier import FaceIdentifier
from api.sample import FacesDatabase
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Get database configuration from environment variables
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PORT = os.getenv("DB_PORT")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

device = "CPU"
faceDETECT = "models/face-detection-retail-0005.xml"
faceLANDMARK = "models/landmarks-regression-retail-0009.xml"
faceINDENTIFY = "models/face-reidentification-retail-0095.xml"

last_insert_time = {}

# Database connection function
def connect_to_db():
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        port=DB_PORT,
        password=DB_PASSWORD,
        database=DB_NAME
    )

class FrameProcessor:
    QUEUE_SIZE = 16

    def __init__(self):
        core = Core()
        self.face_detector = FaceDetector(core, faceDETECT, input_size=(0, 0), confidence_threshold=0.6)
        self.landmarks_detector = LandmarksDetector(core, faceLANDMARK)
        self.face_identifier = FaceIdentifier(core, faceINDENTIFY, match_threshold=0.3, match_algo="HUNGARIAN")
        self.face_detector.deploy(device)
        self.landmarks_detector.deploy(device, self.QUEUE_SIZE)
        self.face_identifier.deploy(device, self.QUEUE_SIZE)
        self.faces_database = FacesDatabase('localhost', "root", 'Face', self.face_identifier, self.landmarks_detector)
        self.face_identifier.set_faces_database(self.faces_database)

    def face_process(self, frame):
        rois = self.face_detector.infer((frame,))
        if self.QUEUE_SIZE > len(rois):
            rois = rois[:self.QUEUE_SIZE]
        landmarks = self.landmarks_detector.infer((frame, rois))
        face_identities, unknowns = self.face_identifier.infer((frame, rois, landmarks))
        return [rois, landmarks, face_identities]

    def draw_face_detection(self, frame, detections):
        size = frame.shape[:2]
        for roi, landmarks, identity in zip(*detections):
            text = self.face_identifier.get_identity_label(identity.id)
            confidence = 1 - identity.distance
            xmin = max(int(roi.position[0]), 0)
            ymin = max(int(roi.position[1]), 0)
            xmax = min(int(roi.position[0] + roi.size[0]), size[1])
            ymax = min(int(roi.position[1] + roi.size[1]), size[0])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 220, 0), 2)
            face_point = xmin, ymin
            for point in landmarks:
                x = int(xmin + roi.size[0] * point[0])
                y = int(ymin + roi.size[0] * point[1])
                cv2.circle(frame, (x, y), 1, (0, 255, 255), 2)
            self.image_recognizer(frame, text, confidence, identity, face_point, 0.75)
        return frame

    def image_recognizer(self, frame, text, confidence, identity, face_point, threshold):
        xmin, ymin = face_point
        display_text = f"{text} ({confidence:.2f})"
        if confidence > threshold:
            textsize = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
            cv2.rectangle(frame, (xmin, ymin), (xmin + textsize[0], ymin - textsize[1]), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, display_text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

            if confidence >= 0.80:
                current_time = time()

                # Check if enough time has passed since the last insertion (300 seconds = 5 minutes)
                if text not in last_insert_time or (current_time - last_insert_time[text] >= 300):
                    face_id = self.faces_database.get_id_by_name(text)  # Get face ID by name

                    if face_id is not None:
                        # Connect to the database
                        db_connection = connect_to_db()
                        cursor = db_connection.cursor()

                        # Prepare the SQL query to insert the data
                        sql = """INSERT INTO log_cv_attendance 
                                 (userid, username, deviceid, extra1, extra2, datetime, created_by, created_on, edited_by, edited_on, blocked)
                                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
                        data = (
                            face_id,
                            text,
                            2,  # Device ID
                            'test',
                            'test123',
                            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'system',  # created_by
                            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # created_on
                            'system',  # edited_by
                            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # edited_on
                            'false'  # blocked
                        )

                        # Execute the query
                        cursor.execute(sql, data)
                        db_connection.commit()
                        print(f"Inserted data for {text}")

                        # Close the connection
                        cursor.close()
                        db_connection.close()

                        # Update the last insertion time
                        last_insert_time[text] = current_time

        else:
            unknown_text = "unknown"
            textsize = cv2.getTextSize(unknown_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
            cv2.rectangle(frame, (xmin, ymin), (xmin + textsize[0], ymin - textsize[1]), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, unknown_text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

frame_processor = FrameProcessor()
cap = cv2.VideoCapture(1)  # Use 0 for the default camera (usually webcam)
while True:
    start_time = perf_counter()
    ret, frame = cap.read()

    if frame is None:
        print("Error: Unable to capture frame from the camera.")
        break

    detections = frame_processor.face_process(frame)
    frame = frame_processor.draw_face_detection(frame, detections)
    cv2.imshow("face recognition Demo", frame)
    key = cv2.waitKey(1)
    if key in {ord('q'), ord("Q"), 27}:
        cv2.destroyAllWindows()
        break
