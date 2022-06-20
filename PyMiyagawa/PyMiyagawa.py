"""
人工知能専門演習ⅠのPython課題にて、私はじゃんけんにおけるマッチングプログラムを作成した。
用いた技術は、HSVにおける色検出と特徴点マッチングである。
まず最初に、じゃんけんの手の形を登録する。この際、肌色検出の検出結果を用いて手の形がわかりやすくなっている。
登録する際、それぞれの手の形の名前の画像を保存する。
次に、どの手に対して勝つのか引き分けか、はたまた負けるのかを表示し、それに則った手を出す。
その後、その手の写真も保存し、正解の手と結果の手を並べたウィンドウを出す。
そして、そのウィンドウ内で特徴店マッチングを行う。
このプログラムにおいて工夫した点は、じゃんけんという誰でも楽しみやすい要素を取り入れた点と、勝ち負けの照合が特徴点マッチングを用いた点である。
至らなかった点は、結果が正解だったのかの判別と日本語テキストの表示である。
"""

import cv2
import numpy as np
import time
from random import randint

cap = cv2.VideoCapture(0)

bg = cv2.imread("bg.png")

i = 0
s = 0
o = 0
words = ['Lose', 'Draw', 'Win', 'Show Rock in rectangle', 'Show Paper in rectangle', 'Show Scissors in rectangle']
word = ""
nums = ["3", "2", "1", ""]
wl = 0

while True:
    ret ,img = cap.read()

    img = cv2.resize(img, dsize = None, fx=1.3, fy = 1.3)

    HSV_MIN = np.array([0, 30, 60])
    HSV_MAX = np.array([20, 150, 255])

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_MIN, HSV_MAX)

    wid = mask.shape[1]
    hei = mask.shape[0]

    time.sleep(0.1)
    i += 1
    if i == 10:
        s += 1
        i = 0

    if s <= 3:
        word = words[3]
        if s == 3:
            cv2.imwrite('rock.png', rect)
            print(rect)
    elif 3 < s <= 7:
        word = words[4]
        if s == 7:
            cv2.imwrite('paper.png', rect)
    elif 7 < s <= 11:
        word = words[5]
        if s == 11:
            cv2.imwrite('scissors.png', rect)
    elif s > 12:
        word = "enemy : " + enemy 
        

    cv2.rectangle(mask, (0, 0), (mask.shape[1]//2, mask.shape[0]//2), (255, 255, 255), 2)
    rect = mask[0: mask.shape[0]//2, 0:mask.shape[1]//2]

    cv2.putText(mask, text=word,org=(0, mask.shape[0]//2+25),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1.0,color=(255, 255, 255),thickness=2,lineType=cv2.LINE_4)
    
    cv2.putText(mask, text=nums[s%4],org=(10, 30),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1.0,color=(255, 255, 255),thickness=2,lineType=cv2.LINE_4)
    
    if s > 11:
        if s % 4 == 0 and i % 10 == 0:
            o = randint(0,2)
            if o == 0:
                enemy = "Rock"
            elif o == 1:
                enemy = "Paper"
            elif o == 2:
                enemy = "Scissors"
            wl = randint(0,2)

        word2 = "You : " + words[wl]
        cv2.putText(mask, text=word2,org=(0, mask.shape[0]//2+55),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1.0,color=(255, 255, 255),thickness=2,lineType=cv2.LINE_4)

        if s % 4 == 3 and i % 10 == 0:
            cv2.imwrite('result.png', rect)
            result = cv2.imread('result.png', cv2.IMREAD_GRAYSCALE)
            if o == 1 and wl == 0 or o == 0 and wl == 1 or o == 2 and wl == 2:
                ans = cv2.imread("rock.png", cv2.IMREAD_GRAYSCALE)
            elif o == 2 and wl == 0 or o == 1 and wl == 1 or o == 0 and wl == 2:
                ans = cv2.imread("paper.png", cv2.IMREAD_GRAYSCALE)
            elif o == 0 and wl == 0 or o == 2 and wl == 1 or o == 1 and wl == 2:
                ans = cv2.imread("scissors.png", cv2.IMREAD_GRAYSCALE)
            
            akaze = cv2.ORB_create()
            kp1, des1 = akaze.detectAndCompute(result, None)
            kp2, des2 = akaze.detectAndCompute(ans, None)

            bf = cv2.BFMatcher()

            matches = bf.knnMatch(des1, des2, k=2)

            ratio = 0.5
            good = []
            for m, n in matches:
                if m.distance < ratio * n.distance:
                    good.append([m])

            resultImg = cv2.drawMatchesKnn(result, kp1, ans, kp2, matches[:30], None, flags=2)
        if s > 15:
            cv2.imshow("result", resultImg)
            
    cv2.imshow("mask", mask)

    cv2.waitKey(30)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()