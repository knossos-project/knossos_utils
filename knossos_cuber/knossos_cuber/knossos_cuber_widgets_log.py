# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'knossos_cuber_widgets_log.ui'
#
# Created by: PyQt5 UI code generator 5.7
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_dialog_log(object):
    def setupUi(self, dialog_log):
        dialog_log.setObjectName("dialog_log")
        dialog_log.resize(400, 300)
        self.plain_text_edit_log = QtWidgets.QPlainTextEdit(dialog_log)
        self.plain_text_edit_log.setGeometry(QtCore.QRect(3, 7, 391, 281))
        self.plain_text_edit_log.setObjectName("plain_text_edit_log")

        self.retranslateUi(dialog_log)
        QtCore.QMetaObject.connectSlotsByName(dialog_log)

    def retranslateUi(self, dialog_log):
        _translate = QtCore.QCoreApplication.translate
        dialog_log.setWindowTitle(_translate("dialog_log", "Cubing Log"))

