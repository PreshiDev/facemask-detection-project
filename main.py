import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from facemaskdesign import Ui_MainWindow
from PyQt5 import QtWidgets

# USAGE
# python detect_mask_video.py

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os


def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            if face.any():
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)

prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
weightsPath = os.path.sep.join(["face_detector",
                                "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model("mask_detector.model")

# Main application window
class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        # Load Ui file
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # Video stream function
        self.updateComponents()

    # Set image to image label
    @pyqtSlot(QImage)
    def setImage(self, image):
        self.ui.imageFrame.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(bool)
    def maskUpdate(self, status):
        if status:
            self.ui.facemasklabel.setText("FaskMask Detected: Yes")
        else:
            self.ui.facemasklabel.setText("FaskMask Detected: No")

    @pyqtSlot(bool)
    def personUpdate(self, status):
        if status:
            self.ui.personlabel.setText("Person Detected: Yes")
        else:
            self.ui.personlabel.setText("Person Detected: No")
    # Run thread to update images and labels
    def updateComponents(self):
        self.wt = WorkerThread(self)
        self.wt.ImageUpdate.connect(self.setImage)
        self.wt.maskUpdate.connect(self.maskUpdate)
        self.wt.personUpdate.connect(self.personUpdate)

        self.wt.start()
        self.show()


# Thread for displaying images and labels
class WorkerThread(QThread):
    ImageUpdate = pyqtSignal(QImage)
    maskUpdate = pyqtSignal(bool)
    personUpdate = pyqtSignal(bool)

    video = cv2.VideoCapture(0)
    def run(self):
        # loop over the frames from the video stream
        while True:
            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 400 pixels

            ret, frame = self.video.read()

            # detect faces in the frame and determine if they are wearing a
            # face mask or not
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

            # loop over the detected face locations and their corresponding
            # locations
            self.personUpdate.emit(False)
            self.maskUpdate.emit(False)
            for (box, pred) in zip(locs, preds):
                # unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                # determine the class label and color we'll use to draw
                # the bounding box and text
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                if label == "Mask":
                    self.maskUpdate.emit(True)
                else:
                    self.maskUpdate.emit(False)

                # include the probability in the label
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                # display the label and bounding box rectangle on the output
                # frame
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                self.personUpdate.emit(True)

            # show the output frame
            #cv2.imshow("Frame", frame)

            # Update Video
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = img.scaled(900, 900, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.ImageUpdate.emit(pixmap)

    def stop(self):
        self.quit()


def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()