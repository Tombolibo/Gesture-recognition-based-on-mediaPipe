import cv2
import numpy as np
import mediapipe as mp

print(cv2.__version__)
print(np.__version__)
print(mp.__version__)


class HandsDetector(object):
    def __init__(self, cameraIndex = 0):
        self._img = None
        self._imgShow = None
        self._cameraIndex = cameraIndex
        self._camera = None
        self._windowName = 'Hand detector'
        self._hands = mp.solutions.hands
        self._detectResult = None
        self._handsModel = None
        self._keyCode = None

        cv2.namedWindow(self._windowName)
        self.openCamera()
        self.handsInit()
        print('init done')

    #初始化摄像头
    def openCamera(self):
        self._camera = cv2.VideoCapture(self._cameraIndex)

    #捕获帧并展示在img中
    def grabImg(self):
        if self._camera is not None:
            ret, self._img = self._camera.read()
            self._imgShow = self._img.copy()

    #初始化手部检测模型
    def handsInit(self):
        self._handsModel = self._hands.Hands()

    #侦测手部
    def detectHands(self):
        imgRGB = cv2.cvtColor(self._img, cv2.COLOR_BGR2RGB)
        self._detectResult = self._handsModel.process(image = imgRGB)
        if self._detectResult.multi_hand_landmarks is not None:
            print('手的个数：', len(self._detectResult.multi_hand_landmarks))
            #获取每只手的某个点的像素坐标
            #对于返回结果中的每只手
            for hand_landmarks in self._detectResult.multi_hand_landmarks:
                # pt_x = int(hand_landmarks.landmark[self._hands.HandLandmark.INDEX_FINGER_TIP].x * imgRGB.shape[1])
                # pt_y = int(hand_landmarks.landmark[self._hands.HandLandmark.INDEX_FINGER_TIP].y * imgRGB.shape[0])
                # for i in range(21):
                #     pt_x = int(hand_landmarks.landmark[i].x * imgRGB.shape[1])
                #     pt_y = int(hand_landmarks.landmark[i].y * imgRGB.shape[0])
                #     cv2.circle(self._imgShow, (pt_x, pt_y), 8, (0,0,255), -1)
                mp.solutions.drawing_utils.draw_landmarks(self._imgShow, hand_landmarks, self._hands.HAND_CONNECTIONS)

    #展示图片
    def showImg(self):
        if self._imgShow is not None:
            cv2.imshow(self._windowName, self._imgShow)


    #运行
    def run(self):
        while self._keyCode != 27:
            self.grabImg()
            self.detectHands()
            self.showImg()
            self._keyCode = cv2.waitKey(10)


if __name__ == '__main__':
    myHandDetector = HandsDetector(0)
    myHandDetector.run()


