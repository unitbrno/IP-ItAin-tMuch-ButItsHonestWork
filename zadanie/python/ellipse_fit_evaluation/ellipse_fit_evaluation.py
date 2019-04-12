import csv

import cv2
import numpy as np


def evaluate_ellipse_fit(image_filename, fit_ellipse, csv_filepath='ground_truths.csv'):
    """
    Evaluate ellipse fit by comparing its parameters to the ground truth ellipse's parameters from the CSV file.
    :param image_filename: filename of the image
    :param fit_ellipse: fitted ellipse's parameters
    :param csv_filepath: filepath to the CSV file with ground truth ellipses' parameters
    :return: fitted ellipse's fit score
    """
    gt_ellipse = __get_gt_ellipse_from_csv(image_filename, csv_filepath)

    if gt_ellipse:
        if fit_ellipse:
            return __get_ellipse_fit_score(fit_ellipse, gt_ellipse)
        else:
            return 0.0
    else:
        if fit_ellipse:
            return 0.0
        else:
            return 1.0


def __get_gt_ellipse_from_csv(image_filename, csv_filepath):
    """
    Return the ground truth ellipse's parameters from the CSV file.
    :param image_filename: filename of the image
    :param csv_filepath: filepath to the CSV file with ground truth ellipses' parameters
    :return: ground truth ellipse's parameters
    """
    with open(csv_filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['filename'] == image_filename:
                if row['gt_ellipse_center_x'] == '':
                    gt_ellipse = None
                else:
                    gt_ellipse = {'center': (float(row['gt_ellipse_center_x']), float(row['gt_ellipse_center_y'])),
                                  'axes': (float(row['gt_ellipse_majoraxis']), float(row['gt_ellipse_minoraxis'])),
                                  'angle': int(row['gt_ellipse_angle']),
                                  'image_width': int(row['image_width']),
                                  'image_height': int(row['image_height'])}
                return gt_ellipse
        else:
            raise ValueError("Filename not found in the CSV file.")


def __get_ellipse_fit_score(fit_ellipse, gt_ellipse):
    """
    Calculate ellipse's fit score based on fitted and ground truth ellipses' parameters.
    :param fit_ellipse: fitted ellipse's parameters
    :param gt_ellipse: ground truth ellipse's parameters
    :return: fitted ellipse's fit score
    """
    fit_ellipse_image = __draw_ellipse(fit_ellipse, (gt_ellipse['image_width'], gt_ellipse['image_width']))
    gt_ellipse_image = __draw_ellipse(gt_ellipse, (gt_ellipse['image_width'], gt_ellipse['image_width']))

    return __evaluate_overlap(gt_ellipse_image, fit_ellipse_image)


def __draw_ellipse(ellipse, image_shape):
    """
    Creates a binary image containing an ellipse.
    :param ellipse: dict with 'center', 'axes' and 'angle' keys
    :param image_shape: (int, int) == (width, height) the dimensions of the image (NumPy shape)
    :return: binary image containing an ellipse
    """
    ellipse_image = np.zeros(image_shape, np.uint8)
    center = (int(np.around(ellipse['center'][0])), int(np.around(ellipse['center'][1])))
    axes = (int(np.around(ellipse['axes'][0])), int(np.around(ellipse['axes'][1])))
    angle = ellipse['angle']

    cv2.ellipse(ellipse_image, center, axes, angle, 0, 360, 1, 1)
    cv2.ellipse(ellipse_image, center, axes, angle, 0, 360, 1, cv2.FILLED)

    return ellipse_image


def __evaluate_overlap(gt_ellipse_image, fit_ellipse_image):
    """
    Calculate overlap of two binary images of ellipses.
    :param gt_ellipse_image: (numpy.ndarray) binary image with the ground truth ellipse
    :param fit_ellipse_image: (numpy.ndarray) binary image with the fitted ellipse
    :return: relative overlap of two binary images of ellipses
    """
    gt_sum = np.sum(gt_ellipse_image)
    fit_sum = np.sum(fit_ellipse_image)

    overlap_image = gt_ellipse_image & fit_ellipse_image
    overlap_sum = np.sum(overlap_image)

    return overlap_sum / max(fit_sum, gt_sum)
