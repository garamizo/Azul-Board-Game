# %% Import images
import azul.vision as vision
import cv2
import numpy as np
import matplotlib.pyplot as plt
%reload_ext autoreload
%autoreload 2


img = cv2.imread('../Pictures/azul/raw/20230831_184400.jpg')  # hard

imgBoards, maskMerge, matrices = vision.get_boards(img, plot=True)
h, w, c = imgBoards[0].shape
ksz = 5

# %%


imgBoards, maskMerge, matrices = vision.get_boards(img, plot=True)
i = 0

plt.subplot(221), plt.imshow(imgBoards[0])
plt.subplot(222), plt.imshow(imgBoards[1])
plt.subplot(223), plt.imshow(imgBoards[2])
plt.subplot(224), plt.imshow(imgBoards[3])
matrices
