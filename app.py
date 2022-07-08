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

BOUNDARYINC = 5
WINDOWSIZEX = 640
WINDOWSIZEY = 480
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
IMAGESAVE = False
iswriting = False
FONT = pygame.font.Font('freesansbold.ttf', 32)
MODEL = load_model("model.h5")
PREDICT = True

LABELS = {0:'ZERO',1:'ONE',
          2:'TWO',3:'THREE',
          4:'FOUR',5:'FIVE',
          6:'SIX',7:'SEVEN',
          8:'EIGHT',9:'NINE'}

pygame.display.set_caption("DigitBoard")
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX,WINDOWSIZEY))

number_xcord = []
number_ycord = []
img_cnt = 1

while(True):
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)

            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False

            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            rect_min_x, rect_max_x = max(number_xcord[0]-BOUNDARYINC, 0), min(WINDOWSIZEX, number_xcord[-1]+BOUNDARYINC)
            rect_min_y, rect_max_y = max(0, number_ycord[0]-BOUNDARYINC ), min(number_ycord[-1]+BOUNDARYINC,WINDOWSIZEX)
            

            number_xcord = []
            number_ycord = []

            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x,rect_min_y:rect_max_y].T.astype(np.float32)

            if IMAGESAVE:
                cv2.imwrite("image.png")
                img_cnt += 1

            if PREDICT:

                img = cv2.resize(img_arr,(28,28))
                img = np.pad(img, (10,10),'constant',constant_values=0)
                img = cv2.resize(img,(28,28))/255

                label = str(LABELS[np.argmax(MODEL.predict(img.reshape(1,28,28,1)))])
                textsurface = FONT.render(label, True, RED, WHITE)
                textrecobj = textsurface.get_rect()
                textrecobj.left, textrecobj.bottom = rect_min_x, rect_max_y
               
                DISPLAYSURF.blit(textsurface,textrecobj)

            if event.type == KEYDOWN:
                if event.unicode == 'n':
                    DISPLAYSURF.fill(BLACK)


        pygame.display.update()