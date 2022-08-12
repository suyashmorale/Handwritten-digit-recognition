from msilib.schema import Font
import matplotlib.pyplot as plt
import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2 
from tensorflow.python.keras.backend import constant
import os


pygame.init()

WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
WINDOWSIZEX = 640
WINDOWSIZEY = 480
iswriting = False
PREDICT = True
FONT = pygame.font.Font('freesansbold.ttf', 32)
BOUNDARYINC = 5
MODEL = load_model("model.h5")
LABELS = {0:'ZERO',1:'ONE',
          2:'TWO',3:'THREE',
          4:'FOUR',5:'FIVE',
          6:'SIX',7:'SEVEN',
          8:'EIGHT',9:'NINE'}

pygame.display.set_caption("DigitBoard")
board = pygame.display.set_mode((WINDOWSIZEX,WINDOWSIZEY))

xcord_list = []
ycord_list = []

while(True):
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()


        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            
        if event.type == MOUSEMOTION and iswriting:
                xcord, ycord = event.pos
                pygame.draw.circle(board, WHITE, (xcord, ycord), 6, 0)
                xcord_list.append(xcord)
                ycord_list.append(ycord)
        
        if event.type == MOUSEBUTTONUP:
            iswriting = False

            if len(xcord_list)==0 or len(ycord_list)==0:
                continue
            else:
                xcord_list = sorted(xcord_list)
                ycord_list = sorted(ycord_list)

                rect_min_x, rect_max_x = max(xcord_list[0]-BOUNDARYINC, 0), min(WINDOWSIZEX, xcord_list[-1]+BOUNDARYINC)
                rect_min_y, rect_max_y = max(0, ycord_list[0]-BOUNDARYINC), min(ycord_list[-1]+BOUNDARYINC,WINDOWSIZEY)

                xcord_list = []
                ycord_list = []
                img_arr = np.array(pygame.PixelArray(board))[rect_min_x:rect_max_x,rect_min_y:rect_max_y].T.astype(np.float32)
            if PREDICT:
                img = cv2.resize(img_arr,(28,28))
                img = np.pad(img, (10,10),'constant',constant_values=0)
                img = cv2.resize(img,(28,28))/255

                label = str(LABELS[np.argmax(MODEL.predict(img.reshape(1,28,28,1)))])
                textsurface = FONT.render(label,True,RED)
                textrecobj = textsurface.get_rect()
                textrecobj.left, textrecobj.top = rect_min_x, rect_max_y
                pygame.draw.rect(board,RED,pygame.Rect(rect_min_x,rect_min_y,rect_max_x-rect_min_x,rect_max_y-rect_min_y),2,1)

                board.blit(textsurface,textrecobj)


    pygame.display.update()


