import cv2
import numpy as np
import mediapipe as mp

print('opencv: ', cv2.__version__)
print('numpy: ', np.__version__)
print('mediaPipe: ', mp.__version__)

print('cv2.version: ', cv2.__version__)
print('np.version: ', np.__version__)
print('mp.version: ', mp.__version__)

class Gesture(object):
    def __init__(self, mode = 0, fileName = None):  #0摄像头，1视频，2图片
        self._mode = mode
        self._filePath = fileName
        self._ret = False
        self._img = None
        self._imgShow = None
        self._camera = None  #摄像头
        self._windowName = 'Gesture Recognition'
        self._keyCode = -1
        self._hands = mp.solutions.hands
        self._handsModel = self._hands.Hands(False, 2, 1, 0.7, 0.5)  #侦测手部模型
        self._handsData = None  #multi_hand_landmarks of model process
        #剪刀石头布胜利者测试
        self._RPSWinner = [0,0]  #测试用
        self._shapes = ['', '']  #当前手势
        self._winningSentence = 'Won'
        self._lossingSentence = 'Lost'
        self._winningSentenceSize = cv2.getTextSize(self._winningSentence, cv2.FONT_HERSHEY_SIMPLEX, 1,2)[0]
        self._lossingSentenceSize = cv2.getTextSize(self._lossingSentence, cv2.FONT_HERSHEY_SIMPLEX, 1,2)[0]
        self._fingerStraight = []  #每只手的每个手指伸直情况

        if self._mode == 0:  # 摄像头模式
            self._camera = cv2.VideoCapture(0)
            print('camera opened: ', self._camera.isOpened())
            self.grabImg()
        elif self._mode == 1:  # 摄像头模式
            self._camera = cv2.VideoCapture(self._filePath)
            print('video readed: ', self._camera.isOpened())
            self.grabImg()
        elif self._mode == 2:
            self._img = cv2.imread(self._filePath)
            self._img = cv2.resize(self._img, (400,300))
            print('图片读取完成: ', self._img.shape)
        self.createWindow()


        self._fontSize = int(np.mean(self._img.shape[:2])//250)
        self._fontThickness = int(np.mean(self._img.shape[:2])//250)

    #创建窗口
    def createWindow(self):
        cv2.namedWindow(self._windowName, cv2.WINDOW_NORMAL)

    #从摄像头或视频获取文件，图片不用，已经在init里面读取了直接
    def grabImg(self):
        self._ret, self._img = self._camera.read()

    #预处理图像,获取手数据
    def preprocessImg(self):
        if self._img is None and self._ret:
            print('img None')
            return
        #将当前摄像头捕捉到的图片转换为mediapipe需要的RGB空间
        imgRGB = cv2.cvtColor(self._img, cv2.COLOR_BGR2RGB)
        handsDetectResults = self._handsModel.process(imgRGB)  #识别手部（可能存在多只手，循环处理）
        self._fingerStraight.clear()  #将上一帧的手指情况清空
        #判断存在多只手的数据
        if handsDetectResults.multi_hand_landmarks is not None:
            self._handsData = handsDetectResults.multi_hand_landmarks  #将多只手数据存储
            #绘制每只手
            for hand_landmarks in handsDetectResults.multi_hand_landmarks:
                handPos = self.getOneHandPos(hand_landmarks)  #计算当前这只手的所有21点x、y像素坐标
                self._fingerStraight.append(self.directionCal(handPos, eps=15))  #通过坐标判断当前手的每个手指是否伸直，并存储
                mp.solutions.drawing_utils.draw_landmarks(self._img, hand_landmarks, self._hands.HAND_CONNECTIONS)  #绘制

    def getOneHandPos(self, hand_landmarks):
        if hand_landmarks is None:
            return
        x = np.zeros((1,21), dtype = np.int32)
        y = np.zeros((1,21), dtype = np.int32)
        for i in range(21):
            x[0][i] = hand_landmarks.landmark[i].x*(self._img.shape[1])
            y[0][i] = hand_landmarks.landmark[i].y*(self._img.shape[0])
        return np.concatenate([x,y], axis = 0)

    #计算两向量夹角
    def vectorAngle(self, v1, v2):
        """计算两个向量之间的夹角（度）"""
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        cos_theta = dot_product / (norm_product + 1e-6)  # 避免除以零
        return np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))

    #计算一只手的五根手指角度，并返回该手的五根手指是否伸直
    def directionCal(self, handPos, eps = 15):
        result = np.zeros((5, ), dtype = np.int32)
        disEps = np.sqrt((handPos[0][0]-handPos[0][9])**2 + (handPos[1][0]-handPos[1][9])**2)
        #大拇指特殊处理
        fingerPos = 1
        dir1 = (handPos[0][fingerPos + 1] - handPos[0][fingerPos + 0],
                handPos[1][fingerPos + 1] - handPos[1][fingerPos + 0])
        dir2 = (handPos[0][fingerPos + 2] - handPos[0][fingerPos + 1],
                handPos[1][fingerPos + 2] - handPos[1][fingerPos + 1])
        dir3 = (handPos[0][fingerPos + 3] - handPos[0][fingerPos + 2],
                handPos[1][fingerPos + 3] - handPos[1][fingerPos + 2])
        theta1 = self.vectorAngle(dir1, dir2)
        theta2 = self.vectorAngle(dir2, dir3)
        #除了角度，还要考虑指尖到掌心距离
        palmDis = np.sqrt((handPos[0][0]-handPos[0][4])**2+(handPos[1][0]-handPos[1][4])**2)
        result[0] = np.abs(theta1) <= 2*eps and np.abs(theta2) < 2.5*eps and palmDis>disEps
        #处理剩余4手指
        for i in range(1,5):
            fingerPos = i*4+1
            dir1 = (handPos[0][fingerPos+1]-handPos[0][fingerPos+0],
                    handPos[1][fingerPos+1]-handPos[1][fingerPos+0])
            dir2 = (handPos[0][fingerPos+2]-handPos[0][fingerPos+1],
                    handPos[1][fingerPos+2]-handPos[1][fingerPos+1])
            dir3 = (handPos[0][fingerPos+3]-handPos[0][fingerPos+2],
                    handPos[1][fingerPos+3]-handPos[1][fingerPos+2])
            theta1 = self.vectorAngle(dir1, dir2)
            theta2 = self.vectorAngle(dir2, dir3)
            fingerDis = np.sqrt((handPos[0][0]-handPos[0][i*4+4])**2+(handPos[1][0]-handPos[1][i*4+4])**2)
            # result[i] = np.abs(theta1-theta2)<=eps and np.abs(theta2-theta3)<eps and fingerDis>disEps
            result[i] = np.abs(theta2)<eps and fingerDis>disEps*1.2
        return result

    #判断是石头、剪刀、还是布，返回：'rock', 'paper', 'scissors'
    def rockPaperScissor(self, fingerData):
        if np.sum(fingerData) == 0:
            return 'rock'
        elif fingerData[0] == 0 and fingerData[1]==1 and fingerData[2]==1 and fingerData[3]==0 and fingerData[4]==0:
            return 'scissors'
        elif np.sum(fingerData)==5:
            return 'paper'
        elif fingerData[0]==0 and fingerData[1]==0 and fingerData[2]==1 and fingerData[3]==0 and fingerData[4]==0:
            return 'Don\'t (`0_0)'
        else:
            return ''

    #判断剪刀石头布谁胜利
    def findWinner(self, hands):
        if len(hands) == 2:
            firstHand = self.rockPaperScissor(hands[0])
            secondHand = self.rockPaperScissor(hands[1])
            if firstHand == 'rock':
                if secondHand == 'rock':
                    self._RPSWinner[0] = 0
                    self._RPSWinner[1] = 0
                elif secondHand == 'paper':
                    self._RPSWinner[0] = -1
                    self._RPSWinner[1] = 1
                elif secondHand == 'scissors':
                    self._RPSWinner[0] = 1
                    self._RPSWinner[1] = -1

            elif firstHand == 'paper':
                if secondHand == 'rock':
                    self._RPSWinner[0] = 1
                    self._RPSWinner[1] = -1
                elif secondHand == 'paper':
                    self._RPSWinner[0] = 0
                    self._RPSWinner[1] = 0
                elif secondHand == 'scissors':
                    self._RPSWinner[0] = -1
                    self._RPSWinner[1] = 1

            elif firstHand == 'scissors':
                if secondHand == 'rock':
                    self._RPSWinner[0] = -1
                    self._RPSWinner[1] = 1
                elif secondHand == 'paper':
                    self._RPSWinner[0] = 1
                    self._RPSWinner[1] = -1
                elif secondHand == 'scissors':
                    self._RPSWinner[0] = 0
                    self._RPSWinner[1] = 0



    def showImg(self):
        self._imgShow = np.ones(self._img.shape, dtype=np.uint8)*255
        w = self._imgShow.shape[1]
        h = self._imgShow.shape[0]
        self._RPSWinner[0] = 0
        self._RPSWinner[1] = 0
        self._shapes[0] = ''
        self._shapes[1] = ''
        self.findWinner(self._fingerStraight)
        for i in range(len(self._fingerStraight)):
            try:
                numPoint = (w // len(self._fingerStraight) * i + (w // len(self._fingerStraight)) // 2, h // 4 * 1)
                textPoint = (w // len(self._fingerStraight) * i + (w // len(self._fingerStraight)) // 2, h // 2)
                winlosePoint = (w // len(self._fingerStraight) * i + (w // len(self._fingerStraight)) // 2, h // 4 * 3)
                numSize = cv2.getTextSize(str(np.sum(self._fingerStraight[i])), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                cv2.putText(self._imgShow, str(np.sum(self._fingerStraight[i])),(numPoint[0]-numSize[0]//2, numPoint[1]-numSize[1]//2),
                            cv2.FONT_HERSHEY_SIMPLEX, self._fontSize,(0,0,0), self._fontThickness)

                #放剪刀石头布
                if self.rockPaperScissor(self._fingerStraight[i]) != '':
                    self._shapes[i] = self.rockPaperScissor(self._fingerStraight[i])
                    textSize = cv2.getTextSize(self._shapes[i], cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    cv2.putText(self._imgShow, self._shapes[i],(textPoint[0]-textSize[0]//2, textPoint[1]-textSize[1]//2),
                                cv2.FONT_HERSHEY_SIMPLEX, self._fontSize,(255,0,0), self._fontThickness)

                #胜利失败的话
                if self._RPSWinner[i] == 1:
                    cv2.putText(self._imgShow, self._winningSentence, (winlosePoint[0]-self._winningSentenceSize[0]//2, winlosePoint[1]-self._winningSentenceSize[1]//2),
                                cv2.FONT_HERSHEY_SIMPLEX, self._fontSize, (0,255,0), self._fontThickness)
                if self._RPSWinner[i] == -1:
                    cv2.putText(self._imgShow, self._lossingSentence, (winlosePoint[0]-self._lossingSentenceSize[0]//2, winlosePoint[1]-self._lossingSentenceSize[1]//2),
                                cv2.FONT_HERSHEY_SIMPLEX, self._fontSize, (0,0,255), self._fontThickness)
            except:
                print('侦测异常，手数量：', len(self._fingerStraight))


        if len(self._fingerStraight) == 0:
            noFingerSize = cv2.getTextSize('no finger', cv2.FONT_HERSHEY_SIMPLEX, 2,3)[0]
            cv2.putText(self._imgShow, 'no finger', (w//2-noFingerSize[0]//2, h//2-noFingerSize[1]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, self._fontSize, (0,0,255), self._fontThickness)
        self._img = cv2.flip(self._img, 1)
        self._imgShow = np.concatenate([self._img, self._imgShow], axis = 1)
        cv2.imshow(self._windowName, self._imgShow)
        if self._mode==0 or self._mode==1:
            self._keyCode = cv2.waitKey(10)
        elif self._mode==2:
            self._keyCode = cv2.waitKey(0)


    #运行
    def run(self):
        if self._mode==0:
            print('摄像头模式')
            while self._keyCode != 27 and cv2.getWindowProperty(self._windowName, cv2.WND_PROP_VISIBLE):
                self.preprocessImg()
                self.showImg()
                self.grabImg()
        elif self._mode==1:
            print('视频模式')
            while self._ret and self._keyCode != 27 and cv2.getWindowProperty(self._windowName, cv2.WND_PROP_VISIBLE):
                self.preprocessImg()
                self.showImg()
                self.grabImg()
        elif self._mode == 2:
            print('图片模式')
            self.preprocessImg()
            self.showImg()

        cv2.destroyAllWindows()


    #打开摄像头获取一帧

if __name__ == '__main__':

    '''
    mode = 0: 摄像头模式，不需要文件名
    mode = 1: 视频模式，传入视频路径
    mode = 2: 图片模式，传入图片路径
    '''

    ges = Gesture(mode=0, fileName=r'./handMoving.mp4')
    ges.run()