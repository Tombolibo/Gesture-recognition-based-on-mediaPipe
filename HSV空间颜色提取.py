import cv2
import numpy as np

# 打开摄像头
cam = cv2.VideoCapture(0)
print(cam.isOpened())

lh = 0
hh = 0
hs = 0
ls = 0
lv = 0
hv = 0

def setlh(pos):
    global lh
    lh = pos
def sethh(pos):
    global hh
    hh = pos
def setls(pos):
    global ls
    ls = pos
def seths(pos):
    global hs
    hs = pos
def setlv(pos):
    global lv
    lv = pos
def sethv(pos):
    global hv
    hv = pos
cv2.namedWindow('test')
cv2.createTrackbar('lh', 'test', 0, 180, setlh)
cv2.createTrackbar('hh', 'test', 180, 180, sethh)
cv2.createTrackbar('ls', 'test', 0, 255, setls)
cv2.createTrackbar('hs', 'test', 255, 255, seths)
cv2.createTrackbar('lv', 'test', 0, 255, setlv)
cv2.createTrackbar('hv', 'test', 255, 255, sethv)



#转换到HSV空间提取手部
def getPart(img, lh,hh, ls,hs, lv,hv):
    # print(lh, hh, ls, hs, lv, hv)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(imgHSV, (lh, lv, ls), (hh, hv, hs))

    #形态学+滤波去噪
    # k = np.ones((10,10), dtype = np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, k, None, (-1,-1), 2)
    # mask = cv2.medianBlur(mask, 7)
        
    imgHand = cv2.bitwise_and(img, img, mask = mask)
    return imgHand

    imgtemp  = cv2.bitwise_and(img, img, mask = mask)
    imgtemp = cv2.medianBlur(imgShow, 7)
    imgtemp = cv2.morphologyEx(imgShow, cv2.MORPH_CLOSE, k, None, (-1, -1), 2)
    return imgtemp


keyCode = 0
while keyCode!= 27:
    cam.grab()
    ret, img = cam.retrieve()
    imgShow = getPart(img, lh, hh, ls, hs, lv, hv)

    imgShow = np.concatenate([img, imgShow], axis = 1)
    imgShow = cv2.flip(imgShow, 1)
    cv2.imshow('test', imgShow)
    keyCode = cv2.waitKey(24)