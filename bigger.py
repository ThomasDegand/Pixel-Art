from numba import jit, cuda
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

@jit
def boost(img):
    p, q, m = np.shape(img)
    imgRes = np.zeros((2*p, 2*q, m), dtype=np.uint8)
    for i in range(2*p):
        for j in range(2*q):
            imgRes[i, j, :] = img[i//2, j//2, :]
    return imgRes

namee = input("Nom de l'image? ")
imgSave = np.array(Image.open(namee))
imgBoost = boost(imgSave)
imgPIL = Image.fromarray(imgBoost)
imgPIL.save(namee[:-4]+"_x2.png")
