import logging as log  # Importing logging module for logging messages
import os  # Importing os module for operating system related functions
import os.path as osp  # Importing os.path module for operating system path manipulations

import cv2  # Importing OpenCV for image processing
import numpy as np  # Importing NumPy for numerical computations
from scipy.optimize import linear_sum_assignment  # Importing linear_sum_assignment function for Hungarian algorithm
from scipy.spatial.distance import cosine  # Importing cosine distance function from scipy.spatial.distance

from api.face_detector import FaceDetector  # Importing FaceDetector class
import mysql.connector  # Importing MySQL connector for database connection


class FacesDatabase:
    IMAGE_EXTENSIONS = ['jpg', 'png']  # Supported image file extensions

    class Identity:
        def __init__(self, label, descriptors):
            self.label = label  # Label for the identity
            self.descriptors = descriptors  # Facial descriptors for the identity

        @staticmethod
        def cosine_dist(x, y):
            return cosine(x, y) * 0.5  # Method to calculate cosine distance between two descriptors

    def __init__(self, host, user, database, face_identifier, landmarks_detector, face_detector=None, no_show=False):
        # Initialize the database connection
        self.conn = mysql.connector.connect(
            host=host,
            user=user,
            database=database
        )
        self.cursor = self.conn.cursor()

        self.database = []  # Initialize database to store identities

        # Retrieve labels from the database
        self.cursor.execute("SELECT name FROM images")
        labels = [row[0] for row in self.cursor.fetchall()]

        for label in labels:
            # Read image from the database
            image = self.read_image(label)
            
            orig_image = image.copy()

            if face_detector:
                # Detect faces in the image using face detector
                rois = face_detector.infer((image,))
                if len(rois) < 1:
                    log.warning("Not found faces on the image '{}'".format(label))
            else:
                # If face detector is not provided, assume the whole image is a face
                w, h = image.shape[1], image.shape[0]
                rois = [FaceDetector.Result([0, 0, 0, 0, 0, w, h])]

            for roi in rois:
                r = [roi]
                # Detect facial landmarks
                landmarks = landmarks_detector.infer((image, r))

                # Start asynchronous processing of face identifier
                face_identifier.start_async(image, r, landmarks)
                # Get descriptors for the face
                descriptor = face_identifier.get_descriptors()[0]

                if face_detector:
                    # Check if face already exists in the database
                    mm = self.check_if_face_exist(descriptor, face_identifier.get_threshold())
                    if mm < 0:
                        crop = orig_image[int(roi.position[1]):int(roi.position[1]+roi.size[1]),
                               int(roi.position[0]):int(roi.position[0]+roi.size[0])]
                        #name = self.ask_to_save(crop)
                        #self.dump_faces(crop, descriptor, name)
                else:
                    # Add face to the database
                    log.debug("Adding label {} to the gallery".format(label))
                    self.add_item(descriptor, label)

    # Method to read image from the database
    def read_image(self, label):
        self.cursor.execute("SELECT image FROM images WHERE name=%s", (label,))
        image_data = self.cursor.fetchone()
        if image_data:
            nparr = np.frombuffer(image_data[0], np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        else:
            return None

    # Method to match faces using descriptors
    def match_faces(self, descriptors, match_algo='HUNGARIAN'):
        database = self.database
        distances = np.empty((len(descriptors), len(database)))
        for i, desc in enumerate(descriptors):
            for j, identity in enumerate(database):
                dist = []
                for id_desc in identity.descriptors:
                    dist.append(FacesDatabase.Identity.cosine_dist(desc, id_desc))
                distances[i][j] = dist[np.argmin(dist)]

        matches = []
        # if user specify MIN_DIST for face matching, face with minium cosine distance will be selected.
        if match_algo == 'MIN_DIST':
            for i in range(len(descriptors)):
                id = np.argmin(distances[i])
                min_dist = distances[i][id]
                matches.append((id, min_dist))
        else:
            # Find best assignments using Hungarian algorithm
            _, assignments = linear_sum_assignment(distances)
            for i in range(len(descriptors)):
                if len(assignments) <= i:  # assignment failure, too many faces
                    matches.append((0, 1.0))
                    continue

                id = assignments[i]
                distance = distances[i, id]
                matches.append((id, distance))

        return matches

    # Method to create a new label for a face
    def create_new_label(self, path, id):
        while osp.exists(osp.join(path, "face{}.jpg".format(id))):
            id += 1
        return "face{}".format(id)

    # Method to check if a face already exists in the database
    def check_if_face_exist(self, desc, threshold):
        match = -1
        for j, identity in enumerate(self.database):
            dist = []
            for id_desc in identity.descriptors:
                dist.append(FacesDatabase.Identity.cosine_dist(desc, id_desc))
            if dist[np.argmin(dist)] < threshold:
                match = j
                break
        return match

    # Method to check if a label already exists in the database
    def check_if_label_exists(self, label):
        match = -1
        import re
        name = re.split(r'-\d+$', label)
        if not len(name):
            return -1, label
        label = name[0].lower()

        for j, identity in enumerate(self.database):
            if identity.label == label:
                match = j
                break
        return match, label

    # Method to add an item (face) to the database
    def add_item(self, desc, label):
        match = -1
        if not label:
            label = self.create_new_label(self.fg_path, len(self.database))
            log.warning("Trying to store an item without a label. Assigned label {}.".format(label))
        else:
            match, label = self.check_if_label_exists(label)

        if match < 0:
            self.database.append(FacesDatabase.Identity(label, [desc]))
        else:
            self.database[match].descriptors.append(desc)
            log.debug("Appending new descriptor for label {}.".format(label))

        return match, label

    # Method to get ID by name
    def get_id_by_name(self, name):
        self.cursor.execute("SELECT id FROM images WHERE name=%s", (name,))
        result = self.cursor.fetchone()
        if result:
            return result[0]  # Return the ID
        return None  # Return None if no ID is found

    def __getitem__(self, idx):
        return self.database[idx]

    def __len__(self):
        return len(self.database)
