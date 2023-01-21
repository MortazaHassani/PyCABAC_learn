import unittest
import random
import cabac
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import cv2

def showimg (vector):
        image = np.reshape(vector[0:128], (4, 32))
        plt.imshow(image, cmap='gray')
        plt.show()

def showimg_0 (vector):
        image = np.reshape(vector[0:1000], (8, 125))
        plt.imshow(image, cmap='gray')
        plt.show()

def plot_all (vector1,vector2,img3,vector4):
    # Create a 2x2 grid of plots
    fig, axes = plt.subplots(nrows=2, ncols=2)
    img1 = np.reshape(vector1[0:], (400, 400))
    img4 = np.reshape(vector4[0:], (400, 400))
    #_, binarized_image = cv2.threshold(img3, 128, 255, cv2.THRESH_BINARY)
    img2 = np.reshape(vector2[0:], (21, 267))
    # Plot data on the first subplot
    axes[0, 0].imshow(img1, cmap='gray')
    axes[0, 0].set_title("encode")

    # Plot data on the second subplot
    axes[0, 1].imshow(img2, cmap='gray')
    axes[0, 1].set_title("bitstream")

    # Plot data on the third subplot
    axes[1, 0].imshow(img3, cmap='gray')
    axes[1, 0].set_title("actual image")

    # Plot data on the fourth subplot
    axes[1, 1].imshow(img4, cmap='gray')
    axes[1, 1].set_title("Decode")

    plt.show()

p1_init = 0.6
shift_idx = 8

image = cv2.imread("D:\GITHUB\wfilter_1.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Could not load image")
else:
    # Binarize the image using a global threshold of 128
    print("image loaded")
    _, binarized_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    binarized_image = np.where(binarized_image>0,1,0)

    # cv2.imshow("Binarized Image", binarized_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
# bitsToEncode = [random.randint(0, 1) for x_ in range(0, 160000)]
bitsToEncode = np.reshape(binarized_image, (1, binarized_image.shape[0]*binarized_image.shape[1]))
bitsToEncode = [i for i in bitsToEncode[0]]

enc = cabac.cabacEncoder()
enc.initCtx([(p1_init, shift_idx), (p1_init, shift_idx)])
enc.start()

for i, bit in enumerate(bitsToEncode):
    if i != 0:
        ctx = bitsToEncode[i - 1]
    else:
        ctx = 0
    enc.encodeBin(bit, ctx)
enc.encodeBinTrm(1)
enc.finish()
enc.writeByteAlignment()
bs = enc.getBitstream()


print("bitstream",len(bs))

#Decoer Section
decodedBits = []
dec = cabac.cabacDecoder(bs)
dec.initCtx([(p1_init, shift_idx), (p1_init, shift_idx)])
dec.start()

for i in range(0, len(bitsToEncode)):
    if i != 0:
        ctx = decodedBits[i - 1]
    else:
        ctx = 0
    decodedBit = dec.decodeBin(ctx)
    decodedBits.append(decodedBit)
dec.decodeBinTrm()
dec.finish()
print(len(bitsToEncode))
if(decodedBits == bitsToEncode):
    print("same")


plot_all(bitsToEncode,bs,image,decodedBits)

