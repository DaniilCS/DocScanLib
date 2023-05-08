import cv2
import numpy as np


def extract_frames(video_path: str, extract_frame_number: int):
    """
    :param video_path: the path to the video file
    :param extract_frame_number: the number of the frame to extract
    :return: tuple with count of extracted frames and array of frames
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception(f'Failed to open video file {video_path}')

    frames = []
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    frame_count = 0
    extracted_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % extract_frame_number == 0:
            extracted_frame_count += 1
            sharpened_image = cv2.filter2D(frame, -1, kernel)
            frames.append(sharpened_image)

        frame_count += 1

    return extracted_frame_count, frames
