import numpy as np
import cv2
from matplotlib import pyplot as plt

imgname = "sea.jpg" # specify image name here
img = cv2.imread("./input-images/" + imgname) 
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (750,350,185,115) # for automated: (1,1,img.shape[1],img.shape[0])

cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask_fg = np.where((mask==2)|(mask==0),0,1).astype('uint8')

img_masked = img*mask_fg[:,:,np.newaxis] 
#plt.imshow(img_masked),plt.colorbar(),plt.show()
cv2.imwrite("./grabcut-output/masked_" + imgname, img_masked)

img_fg = np.where(img_masked > 0, 255, img_masked)
cv2.imwrite("./grabcut-output/fgmask_" + imgname, img_fg)

img_bg = 255 - img_fg
cv2.imwrite("./grabcut-output/bgmask_" + imgname, img_bg)
