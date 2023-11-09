# # import PIL
# # import keras.models
# # import numpy as np
# # from PIL import Image
# # import pygame
# # from pygame.locals import *
# # import os
# # from keras.models import load_model
# #
# # MODEL = keras.models.load_model("C://Users//Yonatan//PycharmProjects//Handwriting2//bestmodel.h5")
# #
# # pygame.init()
# #
# # HEIGHT = 28
# # WIDTH = 28
# # screen = pygame.display.set_mode((28 ,28))
# #
# # pygame.display.set_caption("digit board")
# # isWriting = False
# #
# # loop = True
# #
# # while loop:
# #     for event in pygame.event.get():
# #         if event.type == pygame.MOUSEBUTTONDOWN:
# #             isWriting = True
# #
# #         if event.type == pygame.MOUSEBUTTONUP and isWriting:
# #             pygame.image.save(screen, 'screen.png')
# #             loop = False
# #
# # if os.path.isfile("screen.png.png"):
# #     image = PIL.Image.open("screen.png")
# #     image = PIL.ImageOps.grayscale(image)
# #     # image.show()
# #     # image = np.resize(image)
# #     img = np.array(image)
# #     img = img.astype(np.float32)/255
# #     img = np.expand_dims(img, -1)
# #     img = np.expand_dims(img, 0)
# #     prediction = np.argmax(MODEL.predict(img))
# #     print(prediction)
# #     os.remove("screen.png")
import keras.models
import pygame
import sys
import time
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

MODEl = load_model("bestmodel.h5")

axisX = 640
axisY = 480

PREDICT = True

num_xcord = []
num_ycord = []

img_cnt = 0

pygame.init()

IMAGESAVE = False

DISPLAYSURF = pygame.display.set_mode((axisX, axisY))
pygame.display.set_caption("digit board")

isWriting = False

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEMOTION and isWriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)

            num_xcord.append(xcord)
            num_ycord.append(ycord)

        if event.type == pygame.MOUSEBUTTONDOWN:
            isWriting = True

        if event.type == pygame.MOUSEBUTTONUP:
            isWriting = False
            num_xcord = sorted(num_xcord)
            num_ycord = sorted(num_ycord)

            rect_min_x, rect_max_x = max(num_xcord[0]-5, 0), min(axisX, num_xcord[-1]+5)
            rect_min_y, rect_max_y = max(num_ycord[0] - 5, 0), min(num_ycord[-1] + 5, axisX)

            num_xcord = []
            num_ycord = []

            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)

            if IMAGESAVE:
                cv2.imwrite("img1.png")
                img_cnt += 1

            if PREDICT:
                image = cv2.resize(img_arr, (28, 28))
                image = np.pad(image, (10, 10), 'constant', constant_values=0)
                image = cv2.resize(image, (28, 28))/255

                prediction = np.argmax(MODEl.predict(image.reshape(1, 28, 28, 1)))

                print(prediction)

            if event.type == pygame.KEYDOWN:
                if event.unicode == "n":
                    DISPLAYSURF.fill((0, 0, 0))

        pygame.display.update()




