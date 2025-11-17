import sys
import argparse
import logging
import pathlib
import json

import cv2
import numpy as np

from .blur_detection import estimate_blur
from .blur_detection import fix_image_size
from .blur_detection import pretty_blur_map


def parse_args():
    parser = argparse.ArgumentParser(description='run blur detection on a single image')
    parser.add_argument('-i', '--images', type=str, nargs='+', required=True, help='directory of images')
    parser.add_argument('-s', '--save-path', type=str, default=None, help='path to save output')

    parser.add_argument('-t', '--threshold', type=float, default=100.0, help='blurry threshold')
    parser.add_argument('-f', '--variable-size', action='store_true', help='fix the image size')

    parser.add_argument('-v', '--verbose', action='store_true', help='set logging level to debug')
    parser.add_argument('-d', '--display', action='store_true', help='display images')

    return parser.parse_args()


def blur_binary_mask_from_laplacian(blur_map: np.ndarray,
                                    smooth_sigma: float = 1.0) -> np.ndarray:
    """
    Returns a binary mask with shape = blur_map.shape, dtype=np.uint8, values {0,1}.
    1 = blurry (low |Laplacian|), 0 = sharp.
    """
    x = np.abs(blur_map).astype(np.float32)
    if smooth_sigma > 0:
        x = cv2.GaussianBlur(x, (0, 0), smooth_sigma)

    # Normalize to [0,255], invert so blur becomes high
    x = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)
    inv = 255 - x.astype(np.uint8)

    # Otsu threshold → binary 0/255
    _, bin255 = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Return {0,1}
    return (bin255 // 255).astype(np.uint8)

def save_binary_mask(mask01: np.ndarray, path: str = "blur_mask.png") -> None:
    """Save 0/1 mask as a 0/255 PNG."""
    cv2.imwrite(path, (mask01 * 255).astype(np.uint8))

def overlay_mask_on_image(image_bgr: np.ndarray, mask01: np.ndarray,
                          alpha: float = 0.5) -> np.ndarray:
    """
    Visual debug overlay: paint blurry pixels in red over the original image.
    """
    overlay = image_bgr.copy()
    red = np.zeros_like(image_bgr)
    red[..., 2] = 255  # BGR -> Red
    overlay = cv2.addWeighted(overlay, 1.0, red, alpha, 0)
    # Keep overlay only where mask==1
    out = image_bgr.copy()
    out[mask01.astype(bool)] = overlay[mask01.astype(bool)]
    return out


def find_images(image_paths, img_extensions=['.jpg', '.png', '.jpeg']):
    img_extensions += [i.upper() for i in img_extensions]

    for path in image_paths:
        path = pathlib.Path(path)

        if path.is_file():
            if path.suffix not in img_extensions:
                logging.info(f'{path.suffix} is not an image extension! skipping {path}')
                continue
            else:
                yield path

        if path.is_dir():
            for img_ext in img_extensions:
                yield from path.rglob(f'*{img_ext}')


def blur_masks_from_numpy_batch(
    imgs: np.ndarray,
    threshold: float = 100.0,
) -> np.ndarray:
    for i in range(imgs.shape[0]):
        img = imgs[i]
        blur_map, _, _ = estimate_blur(img, threshold=threshold)
        mask01 = blur_binary_mask_from_laplacian(blur_map)
        if i == 0:
            masks = np.zeros((imgs.shape[0],) + mask01.shape, dtype=np.uint8)
        masks[i] = mask01
        # save_binary_mask(mask01, f"blur_mask_{i}.png")
    return masks


if __name__ == '__main__':
    assert sys.version_info >= (3, 6), sys.version_info
    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level)

    fix_size = not args.variable_size
    logging.info(f'fix_size: {fix_size}')

    if args.save_path is not None:
        save_path = pathlib.Path(args.save_path)
        assert save_path.suffix == '.json', save_path.suffix
    else:
        save_path = None

    results = []

    for image_path in find_images(args.images):
        image = cv2.imread(str(image_path))
        print("image shape:", image.shape)
        
        if image is None:
            logging.warning(f'warning! failed to read image from {image_path}; skipping!')
            continue

        logging.info(f'processing {image_path}')

        if fix_size:
            image = fix_image_size(image)
        else:
            logging.warning('not normalizing image size for consistent scoring!')
        blur_map, score, blurry = estimate_blur(image, threshold=args.threshold)

        logging.info(f'image_path: {image_path} score: {score} blurry: {blurry}')
        results.append({'input_path': str(image_path), 'score': score, 'blurry': blurry})
        mask01 = blur_binary_mask_from_laplacian(blur_map)  # shape HxW, values {0,1}
        
        save_binary_mask(mask01, "blur_mask.png")

        # Optional visual check:
        overlay = overlay_mask_on_image(image, mask01, alpha=0.4)
        cv2.imwrite("blur_overlay.png", overlay)
        
        if args.display:
            cv2.imshow('input', image)
            cv2.imshow('result', pretty_blur_map(blur_map))

            if cv2.waitKey(0) == ord('q'):
                logging.info('exiting...')
                exit()

    if save_path is not None:
        logging.info(f'saving json to {save_path}')

        with open(save_path, 'w') as result_file:
            data = {'images': args.images, 'threshold': args.threshold, 'fix_size': fix_size, 'results': results}
            json.dump(data, result_file, indent=4)
