import os
import cv2
import numpy as np
import math

# deprecated
def fitting_error(contour):
    # fit ellipse
    bounding_rect = cv2.fitEllipse(contour)

    # fit line
    line = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
    p1 = (line[2][0], line[3][0])
    p2 = (p1[0] + line[0][0], p1[1] + line[1][0])

    # find cumulative difference
    cumulative_diff = 0
    for j in range(len(contour)):
        p0 = contour[j][0]
        top = abs((p2[1] - p1[1]) * p0[0] - (p2[0] - p1[0]) * p0[1] + p2[0] * p1[1] - p2[1] * p1[0])
        bottom = pow((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2, 0.5)
        cumulative_diff += top / bottom
    
    # its a line! exterminate it!
    if cumulative_diff < 25 - math.log(min(bounding_rect[1]) * 8):
        return 100000

    # get its focal points
    a = max(bounding_rect[1]) / 2.0
    b = min(bounding_rect[1]) / 2.0
    f = pow(a**2 - b**2, 0.5)
    v_f = (math.cos(bounding_rect[2]) * f, math.sin(bounding_rect[2]) * f)
    f1 = (bounding_rect[0][0] - v_f[0], bounding_rect[0][1] - v_f[1])
    f2 = (bounding_rect[0][0] + v_f[0], bounding_rect[0][1] + v_f[1])
    f = pow(a**2 - b**2, 0.5)
    a2 = max(bounding_rect[1])
    
    # find cumulative difference
    cumulative_diff = 0
    for j in range(len(contour)):
        v1 = (contour[j][0][0] - f1[0], contour[j][0][1] - f1[1])
        v2 = (contour[j][0][0] - f2[0], contour[j][0][1] - f2[1])
        v1s = pow(v1[0]**2 + v1[1]**2, 0.5)
        v2s = pow(v2[0]**2 + v2[1]**2, 0.5)
        cumulative_diff += abs(v1s + v2s - a2)
    return cumulative_diff

# deprecated
def recursive_contour_divide(contour):
    err = fitting_error(contour)
    if err < 400:
        return contour
    if err > 99999:
        return None
    
    if len(contour) > 60:
        half1 = recursive_contour_divide(contour[:int(len(contour) / 2)])
        half2 = recursive_contour_divide(contour[int(len(contour) / 2):])
        if half1 is not None and half2 is not None:
            return np.concatenate((half1, half2), axis = 0)
        if half1 is not None:
            return half1
        if half2 is not None:
            return half2
    
    return None

def sweet_mother_ellipse(image):
    # base params
    old_rect = ((0, 0), (0, 0), 0)
    reduced_contour = None
    fitted = False

    while not fitted:
        # find contours
        _, contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # windows shit
        #contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # this is how we do it

        # get max length contour
        new_cont = np.array([])
        if len(contours) > 0:
            new_cont = contours[0]
            for i in range(len(contours)-1):
                new_cont = np.concatenate((new_cont, contours[i + 1]), axis=0)

        # no ellipse found
        if len(new_cont) < 5:
            return (None, None)

        # get reduced contour - not anymore - deprecated
        reduced_contour = new_cont
        
        if reduced_contour is None:
            reduced_contour = new_cont

        # fit ellipse
        bounding_rect = cv2.fitEllipse(reduced_contour)

        # draw ellipse
        smaller_rect = ((bounding_rect[0][0], bounding_rect[0][1]), (max(abs(bounding_rect[1][0]) - 5, 0) , max(abs(bounding_rect[1][1]) - 5, 0)), bounding_rect[2])
        cv2.ellipse(image, smaller_rect, 255, -1)

        # check new ellipse
        t = 1.0
        if abs(bounding_rect[0][0] - old_rect[0][0]) < t and abs(bounding_rect[0][1] - old_rect[0][1]) < t and abs(bounding_rect[1][0] - old_rect[1][0]) < t and abs(bounding_rect[1][1] - old_rect[1][1]) < t and abs(bounding_rect[2] - old_rect[2]) < t:
            fitted = True
        old_rect = ((bounding_rect[0][0], bounding_rect[0][1]), (bounding_rect[1][0], bounding_rect[1][1]), bounding_rect[2])

    return reduced_contour, old_rect

def fit_ellipse(original, segmented, file_to_open = None):
    # fit ellipse
    reduced_contour, bounding_rect = sweet_mother_ellipse(segmented)

    if bounding_rect is None:
        # find contours
        #_, contours, _ = cv2.findContours(segmented, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # windows shit
        contours, _ = cv2.findContours(segmented, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # this is how we do it

        # get max length contour
        new_cont = np.array([])
        if len(contours) > 0:
            new_cont = contours[0]
            for i in range(len(contours)-1):
                new_cont = np.concatenate((new_cont, contours[i + 1]), axis=0)

        # no ellipse found
        if len(new_cont) < 5:
            return None

        # get reduced contour
        reduced_contour = recursive_contour_divide(new_cont)

        # its bullshit, i did not hit her, i did not. oh hi mark!
        if reduced_contour is None:
            reduced_contour = new_cont

        # fit ellipse
        bounding_rect = cv2.fitEllipse(reduced_contour)

    # draw ellipse
    test = cv2.cvtColor(np.uint8(np.clip(original, 0, 255)), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(test, reduced_contour, -1, (0,255,0), 3)
    cv2.ellipse(test, bounding_rect, (0, 0, 255), 5)

    # count parameters
    ellipse_center_x = bounding_rect[0][0]
    ellipse_center_y = bounding_rect[0][1]
    ellipse_majoraxis = max(bounding_rect[1]) / 2.0
    ellipse_minoraxis = min(bounding_rect[1]) / 2.0
    ellipse_angle = bounding_rect[2] + 90

    # show image
    if __name__ == '__main__':
        cv2.imshow("result", cv2.resize(test, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA))
        if file_to_open is not None:
            ref = cv2.imread(file_to_open, -1)
            cv2.imshow("reference", cv2.resize(ref, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA))
        cv2.waitKey(0)

    ellipse =  {
      "center": [ellipse_center_x, ellipse_center_y],
      "axes": [ellipse_majoraxis, ellipse_minoraxis],
      "angle": ellipse_angle
    }

    return ellipse


if __name__ == '__main__':
    with open("./data/ground_truths_develop.csv") as f:
        content = f.readlines()

    content = [x.strip() for x in content]

    data = []
    for i, item in enumerate(content):
        if i == 0:
            continue
        parametres = item.split(',')

        image = cv2.imread(os.path.join("./data/images/", parametres[0]), -1)
        image = cv2.imread(os.path.join("./data/images/", parametres[0]), -1)
        image = image * np.uint16(65535.0 / max(image.ravel()))
        image = np.uint8(np.clip(255.0 / 65535.0 * image, 0, 255))
        blur = cv2.bilateralFilter(image, 12, 600, 600)
        ret, thresh = cv2.threshold(blur, 75, 255, 0)

        res = fit_ellipse(image, thresh, os.path.join("./data/ground_truths/", parametres[0][:parametres[0].rfind('.')] + ".png"))
        if res is None:
            print(parametres[0])