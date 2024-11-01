from PyQt5 import QtWidgets, QtGui, QtCore

from UI import Ui_MainWindow

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        # TODO
        self.ui.textEdit_Augmented_Reality.setText('Happy World!')
        self.ui.pushButton_Load_Folder.clicked.connect(self.click_Load_Folder)
        self.ui.pushButton_Load_Image_L.clicked.connect(self.click_Load_Image_L)
        self.ui.pushButton_Load_Image_R.clicked.connect(self.click_Load_Image_R)
        self.ui.pushButton_Find_Corners.clicked.connect(self.click_Find_Corners)
        self.ui.pushButton_Find_Intrinsic.clicked.connect(self.click_Find_Intrinsic)
        self.ui.pushButton_Find_Extrinsic.clicked.connect(self.click_Find_Extrinsic)
        self.ui.pushButton_Find_Distortion.clicked.connect(self.click_Find_Distortion)
        self.ui.pushButton_Show_Result.clicked.connect(self.click_Show_Result)
        self.ui.pushButton_Show_Words_On_Board.clicked.connect(self.click_Show_Words_On_Board)
        self.ui.pushButton_Show_Words_Vertically.clicked.connect(self.click_Show_Words_Vertically)
        self.ui.pushButton_Stereo_Disparity_Map.clicked.connect(self.click_Stereo_Disparity_Map)
        self.ui.pushButton_Load_Image1.clicked.connect(self.click_Load_Image1)
        self.ui.pushButton_Load_Image2.clicked.connect(self.click_Load_Image2)
        self.ui.pushButton_Keypoints.clicked.connect(self.click_Keypoints)
        self.ui.pushButton_Matched_Keypoints.clicked.connect(self.click_Matched_Keypoints)

        
    def click_Load_Folder(self):
        print("Load Folder")
    def click_Load_Image_L(self):
        print("Load Image L")
    def click_Load_Image_R(self):
        print("Load Image R")
    def click_Find_Corners(self):
        print("Find Corners")
    def click_Find_Intrinsic(self):
        print("Find Intrinsic")
    def click_Find_Extrinsic(self):
        print("Find Extrinsic")
        value = self.ui.spinBox_Find_Extrinsic.value()
        print(value)
    def click_Find_Distortion(self):
        print("Find Distortion")
    def click_Show_Result(self):
        print("Show Result")
    def click_Show_Words_On_Board(self):
        print("Show Words On Board")
        word=self.ui.textEdit_Augmented_Reality.toPlainText()
        print(word)
    def click_Show_Words_Vertically(self):
        print("Show Words Vertically")
        word=self.ui.textEdit_Augmented_Reality.toPlainText()
        print(word)
    def click_Stereo_Disparity_Map(self):
        print("Stereo Disparity Map")
    def click_Load_Image1(self):
        print("Load Image1")
    def click_Load_Image2(self):
        print("Load Image2")
    def click_Keypoints(self):
        print("Keypoints")
    def click_Matched_Keypoints(self):
        print("Matched Keypoints")
