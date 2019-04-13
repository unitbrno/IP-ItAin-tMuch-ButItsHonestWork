import os
from os import listdir
from os.path import isfile, join
import sys 
import argparse

import cv2
import csv
import shutil
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


from tools import parse_data, perf_measure
from ellipse_fit_evaluation import evaluate_ellipse_fit, __get_gt_ellipse_from_csv
from fit_ellipse import fit_ellipse

from extracting_inception import create_graph, extract_features, extract_feature
from train_svm import train_svm_classifer, get_model

def get_args():
    """
    method for parsing of arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--mode", action="store", default=["eval", "entry"],
                        help="application mode")
    parser.add_argument('--csv-input', action="store", 
                        help='Text file with line names and transcripts for training.')
    parser.add_argument('--csv-output', action="store", 
                        help='Text file with line names and transcripts for training.')
    parser.add_argument('--images-path', action="store", required=True, 
                        help='Text file with line names and transcripts for training.')
    parser.add_argument('--ground-truths-path', action="store", 
                        help='Text file with line names and transcripts for training.')

    args = parser.parse_args()

    if args.mode == "eval":
        if not os.path.isfile(args.csv_input):
            print("CSV input file doesn't exist.")
            sys.exit(-1)
        if not os.path.isdir(args.images_path):
            print("Images path is not valid.")
            sys.exit(-1)
    elif args.mode == "entry":
        if not os.path.isdir(args.images_path):
            print("Images path is not valid.")

    return args


if __name__ == '__main__':
    args = get_args()

    if args.mode == "eval":
        data = parse_data(args.csv_input, args.images_path, args.ground_truths_path)
        score_sum = 0
        true_values = []
        predicted_values = []
            
        if not os.path.exists("./images_png"):
            os.makedirs("./images_png")
        create_graph("./models/tensorflow_inception_graph.pb")
        model = get_model("./models/model.pkl")

        for i, item in enumerate(data):
            bgr = cv2.cvtColor(item.processed_image, cv2.COLOR_GRAY2BGR)
            cv2.imwrite("./images_png/{}.png" .format(item.filename[:-5]), bgr)
            features = extract_feature(["./images_png/{}.png" .format(item.filename[:-5])])
            prediction = model.predict(features)[0]
            if prediction == 0:
                ellipse = None
            else:
                maxval = max(item.image.ravel())
                new_image =  np.uint8(np.clip(255/maxval * item.image, 0, 255))

                # threshold
                kernel = np.ones((3,3),np.float32)/25
                dst = cv2.filter2D(new_image,-1,kernel)
                tmpImg = cv2.fastNlMeansDenoisingColored(cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR), h = 5, templateWindowSize = 5, searchWindowSize = 15)
                blur = cv2.GaussianBlur(cv2.cvtColor(tmpImg, cv2.COLOR_BGR2GRAY),(5,5),0)

                # blur
                ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                thresh = np.uint8(np.clip(thresh, 0, 255))

                ellipse = fit_ellipse(item.image, thresh)

            score = evaluate_ellipse_fit(item.filename, ellipse, args.csv_input)

            predicted_values.append(prediction)
            true_values.append(item.ellipse)
            score_sum += score

            print("{} - {} - {} - {}" .format(item.filename, (round(score, 2)), prediction, item.ellipse))

        shutil.rmtree('./images_png')
        print("Overall score: {}/{} = {}%" .format((round(score_sum, 2)), len(data), round((score_sum/len(data))*100, 2)))

        dictionary = perf_measure(predicted_values, true_values)
        print("TP", dictionary["TP"])
        print("FP", dictionary["FP"])
        print("TN", dictionary["TN"])
        print("FN", dictionary["FN"])

        plt.bar(list(dictionary.keys()), dictionary.values(), color='g')
        plt.show()
    elif args.mode == "entry":
        filenames = [f for f in listdir(args.images_path) if isfile(join(args.images_path, f))]
        
        if not os.path.exists("./images_png"):
            os.makedirs("./images_png")
        create_graph("./models/tensorflow_inception_graph.pb")
        model = get_model("./models/model.pkl")
        
        with open(args.csv_output, mode='w', newline='') as output_csv:
            csv_writer = csv.writer(output_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['filename', 'ellipse_center_x', 'ellipse_center_y',
                                 'ellipse_majoraxis', 'ellipse_minoraxis', 'ellipse_angle',
                                 'elapsed_time'])

            for i, item in enumerate(filenames):
                image = cv2.imread(os.path.join(args.images_path, item), -1)
                start = timer()
                
                image_processed = image * np.uint16(65535.0 / max(image.ravel()))
                image_processed = cv2.resize(image_processed, (960, 960), interpolation = cv2.INTER_AREA)
                bgr = cv2.cvtColor(image_processed, cv2.COLOR_GRAY2BGR)
                cv2.imwrite("./images_png/{}.png" .format(item[:-5]), bgr)
                features = extract_feature(["./images_png/{}.png" .format(item[:-5])])
                prediction = model.predict(features)[0]
                if prediction == 0:
                    ellipse = None
                else:
                    maxval = max(image.ravel())
                    new_image =  np.uint8(np.clip(255/maxval * image, 0, 255))

                    # threshold
                    kernel = np.ones((3,3),np.float32)/25
                    dst = cv2.filter2D(new_image,-1,kernel)
                    tmpImg = cv2.fastNlMeansDenoisingColored(cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR), h = 5, templateWindowSize = 5, searchWindowSize = 15)
                    blur = cv2.GaussianBlur(cv2.cvtColor(tmpImg, cv2.COLOR_BGR2GRAY),(5,5),0)

                    # blur
                    ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    thresh = np.uint8(np.clip(thresh, 0, 255))

                    ellipse = fit_ellipse(image, thresh)
                os.remove("./images_png/{}.png" .format(item[:-5]))
                end = timer()
                elapsed_time = ((end - start)*1000)
                if ellipse == None:
                    csv_writer.writerow([item, '', '', '', '', '', int(elapsed_time)])
                else:
                    csv_writer.writerow([item, round(ellipse['center'][0], 2), round(ellipse['center'][1], 2),
                                         round(ellipse['axes'][0], 2), round(ellipse['axes'][1], 2),
                                         int(ellipse['angle']), int(elapsed_time)])
            shutil.rmtree('./images_png')
