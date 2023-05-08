import cv2
import numpy as np


def biggest_contour(contours):
    """
    Find contour with the biggest area in it
    :param contours: contours that were found on image
    :return: contour with the biggest area in it
    """
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        peri = cv2.arcLength(i, True)

        approx = cv2.approxPolyDP(i, 0.02 * peri, True)
        if area > max_area and len(approx) == 4:
            biggest = approx
            max_area = area

    return biggest, max_area


def reorder_coordinates(points):
    """
    Reorder coordinates for detection frame
    :param points:
    :return: array with reordered points
    """
    points = np.squeeze(points, axis=1)
    points_new = np.zeros((4, 1, 2), dtype=np.int32)
    coordinates_sum = points.sum(1)

    points_new[0] = points[np.argmin(coordinates_sum)]
    points_new[3] = points[np.argmax(coordinates_sum)]

    coordinates_diff = np.diff(points, axis=1)
    points_new[1] = points[np.argmin(coordinates_diff)]
    points_new[2] = points[np.argmax(coordinates_diff)]

    return points_new


def doc_scan_pipeline(img, width: int, height: int):
    """
    :param img: tensor contains frame to detect document
    :param width: required width of image with document
    :param height: required height of image with document
    :return: if document was found return tensor contains doc else return None
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    img = cv2.resize(img, (width, height))

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)

    img_threshold = cv2.Canny(img_blur, 100, 200, L2gradient=True)

    kernel = np.ones((3, 3))
    img_threshold = cv2.dilate(img_threshold, kernel, iterations=2)

    img_contours = img.copy()
    contours, hierarchy = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image=img_contours, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=5)

    biggest, max_area = biggest_contour(contours)
    if max_area < 0.6 * width * height:
        return None

    biggest = reorder_coordinates(biggest)

    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    img_warp_coloured = cv2.warpPerspective(img, matrix, (width, height))

    dim = (width, height)
    img_warp_coloured = cv2.resize(img_warp_coloured, dim, interpolation=cv2.INTER_AREA)
    return img_warp_coloured


def detect_image_batch(images, width: int, height: int):
    detected = []
    for image in images:
        detected_image = doc_scan_pipeline(image, width, height)
        if not(detected_image is None):
            detected.append(detected_image)
    return detected
