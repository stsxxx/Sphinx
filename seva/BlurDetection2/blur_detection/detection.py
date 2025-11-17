#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy


def fix_image_size(image: numpy.array, expected_pixels: float = 2E6):
    ratio = numpy.sqrt(expected_pixels / (image.shape[0] * image.shape[1]))
    return cv2.resize(image, (0, 0), fx=ratio, fy=ratio)


def estimate_blur(image: numpy.array, threshold: int = 100):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_map = cv2.Laplacian(image, cv2.CV_64F)
    score = numpy.var(blur_map)
    return blur_map, score, bool(score < threshold)


# def pretty_blur_map(blur_map: numpy.array, sigma: int = 5, min_abs: float = 0.5):
#     abs_image = numpy.abs(blur_map).astype(numpy.float32)
#     abs_image[abs_image < min_abs] = min_abs

#     abs_image = numpy.log(abs_image)
#     cv2.blur(abs_image, (sigma, sigma))
#     return cv2.medianBlur(abs_image, sigma)


def pretty_blur_map(blur_map: numpy.ndarray, sigma: int = 5, min_abs: float = 0.5,
                    colormap: int = cv2.COLORMAP_JET) -> numpy.ndarray:
    """
    Returns a BGR uint8 heatmap (ready for cv2.imwrite).
    - Takes abs + log to enhance contrast
    - Properly assigns outputs of OpenCV blurs
    - Normalizes to [0, 255]
    - Applies a colormap
    """
    x = numpy.abs(blur_map).astype(numpy.float32)
    x[x < min_abs] = min_abs
    x = numpy.log(x)

    # Make sure sigma is odd and >=3 for median blur
    if sigma % 2 == 0:
        sigma += 1
    sigma = max(sigma, 3)

    x = cv2.blur(x, (sigma, sigma))         # <-- assign back
    x = cv2.medianBlur(x, sigma)            # <-- assign back

    # Normalize to 8-bit for saving/visualization
    x = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)
    x = x.astype(numpy.uint8)

    # Colorize (BGR) so it looks good out of the box
    heat = cv2.applyColorMap(x, colormap)
    return heat