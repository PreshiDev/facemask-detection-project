# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'fasemask.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setMaximumSize(QtCore.QSize(16777215, 40))
        self.frame.setStyleSheet("background-color: rgb(47, 47, 47);")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.personlabel = QtWidgets.QLabel(self.frame)
        self.personlabel.setStyleSheet("color: rgb(255, 255, 255);")
        self.personlabel.setAlignment(QtCore.Qt.AlignCenter)
        self.personlabel.setObjectName("personlabel")
        self.horizontalLayout_2.addWidget(self.personlabel)
        self.line = QtWidgets.QFrame(self.frame)
        self.line.setMinimumSize(QtCore.QSize(5, 0))
        self.line.setStyleSheet("color: rgb(255, 255, 255);")
        self.line.setLineWidth(5)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.horizontalLayout_2.addWidget(self.line)
        self.facemasklabel = QtWidgets.QLabel(self.frame)
        self.facemasklabel.setStyleSheet("color: rgb(255, 255, 255);")
        self.facemasklabel.setAlignment(QtCore.Qt.AlignCenter)
        self.facemasklabel.setObjectName("facemasklabel")
        self.horizontalLayout_2.addWidget(self.facemasklabel)
        self.verticalLayout.addWidget(self.frame)
        self.imageFrame = QtWidgets.QLabel(self.centralwidget)
        self.imageFrame.setObjectName("imageFrame")
        self.verticalLayout.addWidget(self.imageFrame, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.frame_3 = QtWidgets.QFrame(self.centralwidget)
        self.frame_3.setMaximumSize(QtCore.QSize(16777215, 40))
        self.frame_3.setStyleSheet("background-color: rgb(47, 47, 47);")
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame_3)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.frame_3)
        self.label.setStyleSheet("color: rgb(255, 255, 255);")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.verticalLayout.addWidget(self.frame_3)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.personlabel.setText(_translate("MainWindow", "Person Detected: Loading.."))
        self.facemasklabel.setText(_translate("MainWindow", "FaceMask Detected: Loading.."))
        self.imageFrame.setText(_translate("MainWindow", "Loading..."))
        self.label.setText(_translate("MainWindow", "FaceMask Detection System"))
