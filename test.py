import SiftUtil
import cv2

import numpy as np
img1 = cv2.imread('./imgs/3.png')
img2 = cv2.imread('./imgs/2.jpg')

ctpn_boxes = np.random.random((6,8))
idx = SiftUtil.sift_util(img1,img2,ctpn_boxes)
print(idx)


# c_list = [0]*8
#
# c_list[2]+=1
#
# print(c_list.index(max(c_list)))