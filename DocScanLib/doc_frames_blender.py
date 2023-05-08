import cv2
import numpy as np


def blend_docs(detected_docs):
    count_images = len(detected_docs)
    img = detected_docs[0]

    for i in range(count_images - 1):
        img2 = detected_docs[i + 1]
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        matcher = cv2.BFMatcher()
        matches = matcher.match(des1, des2)

        matches = [m for m in matches if m.distance < 100]

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        affine_transform, _ = cv2.estimateAffine2D(src_pts, dst_pts)
        img1_aligned = cv2.warpAffine(img, affine_transform, (img.shape[1], img.shape[0]))

        cv2.imwrite('affine.jpg', img1_aligned)
        img = cv2.addWeighted(img1_aligned, 0.5, img2, 0.5, 0)

    return img