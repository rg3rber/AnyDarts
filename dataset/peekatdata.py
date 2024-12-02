import argparse
from yacs.config import CfgNode as CN
import pandas as pd
import os.path as osp
import os
import cv2
import numpy as np
from time import time
from annotate import draw, get_dart_scores
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-lp', '--labels-path', default='./labels.pkl')


    args = parser.parse_args()
    data = pd.read_pickle(args.labels_path)
    try:
        print(data.head())
        print(data.info())
    except:
        print("Error: No head found")
        #show me info on the dict data
        for x in data:
            print(x)
