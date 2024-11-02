from PyQt5 import QtWidgets, QtGui, QtCore
import os
from UI import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog, QLabel
import cv2
import numpy as np
class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        # TODO
        self.imgs=[]
        self.imgL=None
        self.imgR=None
        self.chessboard_size = (11, 8)
        self.win_size=(5,5)
        self.zero_zone=(-1,-1)
        self.images_points=[]
        self.object_points=[]
        self.intrinsic_matrix=None
        self.distortion_coefficients=None
        self.rvecs=None
        self.tvecs=None
        self.letter_3d_coordinates={}
        self.text=""
        self.ui.spinBox_Find_Extrinsic.setRange(1,15)
        self.ui.spinBox_Find_Extrinsic.setValue(1)
        self.spinBox_value=1
        self.image1=None
        self.image2=None
        self.ui.spinBox_Find_Extrinsic.valueChanged.connect(self.click_Find_Extrinsic_spinBox)
        # self.ui.textEdit_Augmented_Reality.textChanged.connect(self.on_line_edit_changed)
        self.ui.textEdit_Augmented_Reality.setText('Happy World!')
        self.text=self.ui.textEdit_Augmented_Reality.toPlainText()
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
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if folder_path:
            img_paths = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
            img_paths.sort(key = lambda name: int(name[:-4]), reverse = False)            
            for img_path in img_paths:
                img=cv2.imread(folder_path+'/'+img_path)
                self.imgs.append(img)         
        else:
            print('No images found in the selected folder.')
    def click_Load_Image_L(self):
        imgL_path, _ = QtWidgets.QFileDialog.getOpenFileName()
        self.imgL=cv2.imread(str(imgL_path), cv2.IMREAD_COLOR)
        self.imgL = np.array(self.imgL)
    def click_Load_Image_R(self):
        imgR_path, _ = QtWidgets.QFileDialog.getOpenFileName()
        self.imgR=cv2.imread(str(imgR_path), cv2.IMREAD_COLOR)
        self.imgR = np.array(self.imgR)
    def click_Find_Corners(self):
        gray_imgs=[]
        corners_imgs=[]
        for img in self.imgs:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_imgs.append(gray_img)
        chessboard_size = self.chessboard_size
        win_size = self.win_size
        zero_zone = self.zero_zone
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


        for gray_img in gray_imgs:
            ret, corners = cv2.findChessboardCorners(gray_img, chessboard_size, None)
            if ret:
                corners = cv2.cornerSubPix(gray_img, corners, win_size, zero_zone, criteria)
                img_with_corners = cv2.drawChessboardCorners(gray_img, chessboard_size, corners, ret)
                corners_imgs.append(img_with_corners)
            else:
                print('No corners found in the image.')
        for img in corners_imgs:
            scale_percent = 50  # 縮小為 50%
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)  
            cv2.imshow('Corners', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    def click_Find_Intrinsic(self):
        self.object_points=[]
        self.images_points=[]
        chessboard_size = self.chessboard_size
        win_size = self.win_size
        zero_zone = self.zero_zone
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        for img in self.imgs:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray_img, chessboard_size, None)
            if ret:
                corners = cv2.cornerSubPix(gray_img, corners, win_size, zero_zone, criteria)
                self.images_points.append(corners)

                object_points = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
                object_points[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
                self.object_points.append(object_points)
        # 交換順序，確保 imageSize 格式為 (width, height)
        ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(self.object_points, self.images_points, gray_img.shape[::-1], None, None)
        self.intrinsic_matrix=mtx
        print('Intrinsic Matrix:')
        print(mtx)
        self.distortion_coefficients=dist
        self.rvecs=rvecs
        self.tvecs=tvecs
        




    def click_Find_Extrinsic_spinBox(self,spin_box_value):
        self.spinBox_value=spin_box_value

    def click_Find_Extrinsic(self):
        img_index=self.spinBox_value-1
        object_points=self.object_points[img_index]
        image_points=self.images_points[img_index]

        ret, rvecs, tvecs = cv2.solvePnP(object_points, image_points, self.intrinsic_matrix, self.distortion_coefficients)
        extrinsic_matrix = np.column_stack((cv2.Rodrigues(rvecs)[0], tvecs))
        print('Extrinsic Matrix:')
        print(extrinsic_matrix)

    def click_Find_Distortion(self):
        print("Distortion Coefficients: (K1, K2, P1, P2, K3)")
        print(self.distortion_coefficients)


    def click_Show_Result(self):
        undistorted_imgs=[]
        for img in self.imgs:
            undistorted_img = cv2.undistort(img, self.intrinsic_matrix, self.distortion_coefficients)
            undistorted_imgs.append(undistorted_img)
        
        for img,undistorted_img in zip(self.imgs,undistorted_imgs):
            
            cv2.putText(img, 'Distorted Image', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,cv2.LINE_AA)
            cv2.putText(undistorted_img, 'Undistorted Image', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,cv2.LINE_AA)

            concatenated_img = np.concatenate((img, undistorted_img), axis=1)
            scale_percent = 50  # 縮小為 50%
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            concatenated_img = cv2.resize(concatenated_img, dim, interpolation=cv2.INTER_AREA)
            cv2.imshow('Result', concatenated_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    def load_onboard_txt(self):
        self.letter_3d_coordinates={}
        file_path='./Dataset_CvDl_Hw1/Dataset_CvDl_Hw1/Q2_Image/Q2_db/alphabet_db_onboard.txt'
        file_storage=cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
        self.text=self.ui.textEdit_Augmented_Reality.toPlainText()
        print(file_storage)
        print(self.text)
        # read the letters
        for letter in self.text:
            # print(letter)
            # read the letter
            letter_3d_coordinate = file_storage.getNode(letter).mat().astype(np.float32)
            # reshape the letter
            letter_3d_coordinate = letter_3d_coordinate.reshape(-1, 3)
            # store the letter
            self.letter_3d_coordinates[letter] = letter_3d_coordinate

    def load_vertical_txt(self):
        self.letter_3d_coordinates={}
        file_path='./Dataset_CvDl_Hw1/Dataset_CvDl_Hw1/Q2_Image/Q2_db/alphabet_db_vertical.txt'
        file_storage=cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
        self.text=self.ui.textEdit_Augmented_Reality.toPlainText()

        # read the letters
        for letter in self.text:
            # read the letter
            letter_3d_coordinate = file_storage.getNode(letter).mat().astype(np.float32)
            # reshape the letter
            letter_3d_coordinate = letter_3d_coordinate.reshape(-1, 3)
            # store the letter
            self.letter_3d_coordinates[letter] = letter_3d_coordinate

    def offset_letter(self,index,letter):
        letter_3d_coordinate=self.letter_3d_coordinates[letter].copy()
        offset={
            0:[7,5,0],
            1:[4,5,0],
            2:[1,5,0],
            3:[7,2,0],
            4:[4,2,0],
            5:[1,2,0],
        }
        for i, coordinate in enumerate(letter_3d_coordinate):
            coordinate[0] += offset[index][0]
            coordinate[1] += offset[index][1]
            coordinate[2] += offset[index][2]
            letter_3d_coordinate[i] = coordinate
        return letter_3d_coordinate

    def on_line_edit_changed(self, text):
        print("Line edit text changed:", text)
        self.text = text


    def click_Show_Words_On_Board(self):
        # print("Show Words On Board")
        # word=self.ui.textEdit_Augmented_Reality.toPlainText()
        # print(word)
        if len(self.imgs)==0:
            print('Please load the images first.')
            return
        if len(self.text)==0:
            print('Please input the text first.')
            return
        self.click_Find_Intrinsic()
        self.load_onboard_txt()

        for idx,img in enumerate(self.imgs):
            img=img.copy()
            rvec=self.rvecs[idx]
            tvec=self.tvecs[idx]
            for txt_idx,letter in enumerate(self.text):
                letter_3d_coordinate=self.offset_letter(txt_idx,letter)
                image_points, _ = cv2.projectPoints(letter_3d_coordinate, rvec, tvec, self.intrinsic_matrix, self.distortion_coefficients)

                for i in range(0,len(image_points),2):
                    pt1=(int(image_points[i][0][0]),int(image_points[i][0][1]))
                    pt2=(int(image_points[i+1][0][0]),int(image_points[i+1][0][1]))
                    cv2.line(img,pt1,pt2,(0,0,255),5)
            scale_percent = 50  # 縮小為 50%
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)  
            cv2.imshow('Words On Board', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        

    def click_Show_Words_Vertically(self):
        # print("Show Words Vertically")
        # word=self.ui.textEdit_Augmented_Reality.toPlainText()
        # print(word)
        if len(self.imgs)==0:
            print('Please load the images first.')
            return
        if len(self.text)==0:
            print('Please input the text first.')
            return
        self.click_Find_Intrinsic()
        self.load_vertical_txt()

        for idx,img in enumerate(self.imgs):
            img=img.copy()
            rvec=self.rvecs[idx]
            tvec=self.tvecs[idx]
            for txt_idx,letter in enumerate(self.text):
                letter_3d_coordinate=self.offset_letter(txt_idx,letter)
                image_points, _ = cv2.projectPoints(letter_3d_coordinate, rvec, tvec, self.intrinsic_matrix, self.distortion_coefficients)

                for i in range(0,len(image_points),2):
                    pt1=(int(image_points[i][0][0]),int(image_points[i][0][1]))
                    pt2=(int(image_points[i+1][0][0]),int(image_points[i+1][0][1]))
                    cv2.line(img,pt1,pt2,(0,0,255),5)
            scale_percent = 50  # 縮小為 50%
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)  
            cv2.imshow('Vertical Words On Board', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    def click_Stereo_Disparity_Map(self):
        # print("Stereo Disparity Map")
        if self.imgL is None or self.imgR is None:
            print('Please load the images first.')
            return

        self.imgL_gray = cv2.cvtColor(self.imgL, cv2.COLOR_BGR2GRAY)
        self.imgR_gray = cv2.cvtColor(self.imgR, cv2.COLOR_BGR2GRAY)

        stereo=cv2.StereoBM_create(numDisparities=432, blockSize=25)
        disparity=stereo.compute(self.imgL_gray,self.imgR_gray)

        min_disp = disparity.min()
        max_disp = disparity.max()
        self.disparity_map_normalized=(disparity-min_disp)/(max_disp-min_disp)*255
        scale_percent = 50  # 縮小為 50%
        width = int(self.disparity_map_normalized.shape[1] * scale_percent / 100)
        height = int(self.disparity_map_normalized.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(self.disparity_map_normalized, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow('Disparity Map', img)
        # cv2.imshow('Disparity Map', self.disparity_map_normalized)
        width = int(self.imgL.shape[1] * scale_percent / 100)
        height = int(self.imgL.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(self.imgL, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow("image_L", img)
        # cv2.imshow("image_L", self.imgL)
        width = int(self.imgR.shape[1] * scale_percent / 100)
        height = int(self.imgR.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(self.imgR, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow("image_R", img)
        # cv2.imshow("image_R", self.imgR)
        # cv2.setMouseCallback("image_L", self.check_disparity_value)

    # def check_disparity_value(self, event, x, y, flags, param):
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         # get the depth
    #         disparity_value = self.disparity_map_normalized[y, x]

    #         if disparity_value == 0:
    #             print("Failure case")
    #         else:
    #             # compute the corresponding point
    #             corresponding_x = x - disparity_value

    #             print(f"({corresponding_x}, y={y}), dis={disparity_value}")
    #             image_R_copy = self.imgR.copy()
    #             # mark the corresponding point at image_R
    #             cv2.circle(image_R_copy, (corresponding_x, y), 15, (0, 255, 0), -1)
    #             # scale_percent = 50  # 縮小為 50%
    #             # width = int(image_R_copy.shape[1] * scale_percent / 100)
    #             # height = int(image_R_copy.shape[0] * scale_percent / 100)
    #             # dim = (width, height)
    #             # img = cv2.resize(image_R_copy, dim, interpolation=cv2.INTER_AREA)
    #             cv2.imshow("image_R", image_R_copy)

    #     if event == cv2.EVENT_RBUTTONDOWN:
    #         cv2.destroyAllWindows()
        
        
    def click_Load_Image1(self):
        # print("Load Image1")
        file_path=QFileDialog.getOpenFileName(self, 'Select Image1')
        self.image1=cv2.imread(file_path[0])
    def click_Load_Image2(self):
        # print("Load Image2")
        file_path=QFileDialog.getOpenFileName(self, 'Select Image2')
        self.image2=cv2.imread(file_path[0])
    def click_Keypoints(self):
        # print("Keypoints")
        if self.image1 is None:
            print('Please load the images first.')
            return
        img_gray=cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
        sift=cv2.SIFT_create()
        keypoints, descriptors=sift.detectAndCompute(img_gray, None)

        img_with_keypoints=cv2.drawKeypoints(img_gray, keypoints, None,color=(0,255,0))
        scale_percent = 50  # 縮小為 50%
        width = int(img_with_keypoints.shape[1] * scale_percent / 100)
        height = int(img_with_keypoints.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(img_with_keypoints, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow('Keypoints', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def click_Matched_Keypoints(self):
        # print("Matched Keypoints")
        if self.image1 is None or self.image2 is None:
            print('Please load the images first.')
            return
        img1_gray=cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
        img2_gray=cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)
        sift=cv2.SIFT_create()
        keypoints1, descriptors1=sift.detectAndCompute(img1_gray, None)
        keypoints2, descriptors2=sift.detectAndCompute(img2_gray, None)
        bf=cv2.BFMatcher()
        matches=bf.knnMatch(descriptors1, descriptors2, k=2)
        good_matches=[]
        for m,n in matches:
            if m.distance<0.75*n.distance:
                good_matches.append([m])

        img_matches=cv2.drawMatchesKnn(img1_gray, keypoints1, img2_gray, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        scale_percent = 50  # 縮小為 50%
        width = int(img_matches.shape[1] * scale_percent / 100)
        height = int(img_matches.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(img_matches, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow('Matched Keypoints', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
