import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt


DEBUG = False
EROSION_KERNEL_SIZE = (3, 3)
EROSION_ITERATIONS = 3
DILATE_KERNEL_SIZE = (5, 5)
DILATE_ITERATIONS = 2
LABEL_AREA_THRESHOLD = 140


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--debug', type=bool, default=False)

    return parser.parse_args()


def read_img_color_grayscale(file_name):
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def erode_dilate(img):
    img = cv2.erode(img, np.ones((3, 3), dtype=np.uint8))
    img = cv2.dilate(img, np.ones(DILATE_KERNEL_SIZE, dtype=np.uint8), iterations=DILATE_ITERATIONS)
    img = cv2.erode(img, np.ones(EROSION_KERNEL_SIZE, dtype=np.uint8), iterations=EROSION_ITERATIONS)
    return img


def compute_contours(gray):
    clahe = cv2.createCLAHE(1.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2.5)
    gray = cv2.medianBlur(gray, 3)
    gray = erode_dilate(gray)

    if DEBUG:
        cv2.imwrite('target/edged.jpg', gray)

    _, labeled_img = cv2.connectedComponentsWithAlgorithm(gray, 8, cv2.CV_32S, cv2.CCL_GRANA)
    if DEBUG:
        cv2.imwrite('target/labeled.jpg', labeled_img)

    labels = np.unique(labeled_img)
    labels = labels[labels != 0]
    intermediate_global_mask = np.zeros_like(labeled_img, dtype=np.uint8)
    for idx, label in enumerate(labels):
        mask = np.zeros_like(labeled_img, dtype=np.uint8)
        mask[labeled_img == label] = 255
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hull = []
        for cnt in contours:
            hull.append(cv2.convexHull(cnt, False))
        hull_mask = np.zeros(mask.shape, dtype=np.uint8)
        for i in range(len(contours)):
            hull_mask = cv2.drawContours(hull_mask, hull, i, 255, -1, 8)

        intermediate_global_mask = np.clip(intermediate_global_mask + hull_mask, 0, 255)

    if DEBUG:
        cv2.imwrite('target/mask.jpg', intermediate_global_mask)

    return connected_component_segmentation(intermediate_global_mask)


def random_color():
    color = np.random.randint(0, 255, (3,))
    return color


def connected_component_segmentation(int_global_mask):
    _, connected_components, stats, _ = cv2.connectedComponentsWithStatsWithAlgorithm(int_global_mask, 8, cv2.CV_32S, cv2.CCL_GRANA)
    labels = np.unique(connected_components)
    if labels[0] == 0:
        stats = stats[1:]
    labels = labels[labels != 0]

    out_img = np.zeros(int_global_mask.shape + (3,), dtype=np.uint8)
    vertices = []
    for label, stat in zip(labels, stats):
        if stat[cv2.CC_STAT_AREA] >= LABEL_AREA_THRESHOLD:
            mask = np.zeros_like(connected_components, dtype=np.uint8)
            mask[connected_components == label] = 255
            x, y, h, w = cv2.boundingRect(mask)
            vertices.append((x, y, h, w))
            color = random_color()
            out_img[mask == 255] = color

    if DEBUG:
        cv2.imwrite('target/out_img.jpg', out_img)

    return vertices


def main():
    global DEBUG
    args = parse()

    if args.debug is True:
        DEBUG = True

    img, gray = read_img_color_grayscale(args.input)
    gray = cv2.GaussianBlur(gray, (5, 5), 1)
    if DEBUG:
        cv2.imwrite('target/blurred.jpg', gray)
    vertices = compute_contours(gray)

    for v in vertices:
        color = random_color()
        color = (
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255)
        )
        cv2.rectangle(img, (v[0], v[1]), (v[0] + v[2], v[1] + v[3]), color, 2)
    cv2.imwrite('target/output.jpg', img)


if __name__ == '__main__':
    main()
