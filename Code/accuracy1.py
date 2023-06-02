import numpy as np
import cv2
from PIL import Image
k = 1

# segmentation
seg = Image.open('./Result/Test/segmentation.jpg').convert("L")
seg = np.asarray(seg)
 
# ground truth
gt = Image.open('./Dataset/Test/gt.jpg').convert("L")
gt = np.asarray(gt)
# resize gt to match seg dimensions
gt = cv2.resize(gt, seg.shape[::-1])
dice = np.sum(seg[gt==k])*2.0 / (np.sum(seg) + np.sum(gt))

print("Dice similarity score is {}".format(dice))
