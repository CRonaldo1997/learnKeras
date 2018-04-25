import numpy as np
import cv2
from matplotlib import pyplot as plt


def find_index(kp,ctpn_boxes):
    m,n = ctpn_boxes.shape
    assert n == 8
    count_list = [0]*m

    for point in kp:
        x, y = point.pt
        for i in range(m):
            x0 = ctpn_boxes[i][0]
            #y0 = ctpn_boxes[i][1]

            x1 = ctpn_boxes[i][2]
            y1 = ctpn_boxes[i][3]

            #x2 = ctpn_boxes[i][4]
            y2 = ctpn_boxes[i][5]

            #x3 = ctpn_boxes[i][6]
            #y3 = ctpn_boxes[i][7]

            if x>=x0 and x<=x1 and y>=y1 and y<=y2:
                count_list[i] += 1

    return count_list.index(max(count_list))



def sift_util(tpt_img,tar_img,ctpn_boxes):

    MIN_MATCH_COUNT = 10
    img1= cv2.cvtColor(tpt_img,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(tar_img,cv2.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    index = find_index(kp2,ctpn_boxes)
    return index
