import argparse
from yacs.config import CfgNode as CN
import os.path as osp
import os
from dataloader import get_splits
import cv2
import numpy as np
from time import time
from dataset.annotate import draw, get_dart_scores
import pickle
import json
import dataset.distort as distort


def bboxes_to_xy(bboxes, max_darts=3):
    xy = np.zeros((4 + max_darts, 3), dtype=np.float32)
    for cls in range(5):
        if cls == 0:
            dart_xys = bboxes[bboxes[:, 4] == 0, :2][:max_darts]
            xy[4:4 + len(dart_xys), :2] = dart_xys
        else:
            cal = bboxes[bboxes[:, 4] == cls, :2]
            if len(cal):
                xy[cls - 1, :2] = cal[0]
    xy[(xy[:, 0] > 0) & (xy[:, 1] > 0), -1] = 1
    if np.sum(xy[:4, -1]) == 4:
        return xy
    else:
        xy = est_cal_pts(xy)
    return xy


def est_cal_pts(xy):
    missing_idx = np.where(xy[:4, -1] == 0)[0]
    if len(missing_idx) == 1:
        if missing_idx[0] <= 1:
            center = np.mean(xy[2:4, :2], axis=0)
            xy[:, :2] -= center
            if missing_idx[0] == 0:
                xy[0, 0] = -xy[1, 0]
                xy[0, 1] = -xy[1, 1]
                xy[0, 2] = 1
            else:
                xy[1, 0] = -xy[0, 0]
                xy[1, 1] = -xy[0, 1]
                xy[1, 2] = 1
            xy[:, :2] += center
        else:
            center = np.mean(xy[:2, :2], axis=0)
            xy[:, :2] -= center
            if missing_idx[0] == 2:
                xy[2, 0] = -xy[3, 0]
                xy[2, 1] = -xy[3, 1]
                xy[2, 2] = 1
            else:
                xy[3, 0] = -xy[2, 0]
                xy[3, 1] = -xy[2, 1]
                xy[3, 2] = 1
            xy[:, :2] += center
    else:
        # TODO: if len(missing_idx) > 1
        print('Missed more than 1 calibration point')
    return xy

def predict(
        yolo,
        cfg,
        labels_path='./dataset/labels.pkl',
        dataset='d1',
        split='val',
        max_darts=3,
        write=False):
    
    print("failcases are: " + args.fail_cases.__str__())

    np.random.seed(0)

    write_dir = osp.join('./models', cfg.model.name, 'preds', split)
    if write:
        os.makedirs(write_dir, exist_ok=True)

    data = get_splits(labels_path, dataset, split)
    img_prefix = osp.join(cfg.data.path, 'cropped_images', str(cfg.model.input_size))
    img_paths = [osp.join(img_prefix, folder, name) for (folder, name) in zip(data.img_folder, data.img_name)]

    xys = np.zeros((len(data), 7, 3))  # third column for visibility
    data.xy = data.xy.apply(np.array)
    for i, _xy in enumerate(data.xy):
        xys[i, :_xy.shape[0], :2] = _xy
        xys[i, :_xy.shape[0], 2] = 1
    xys = xys.astype(np.float32)

    preds = np.zeros((len(img_paths), 4 + max_darts, 3))
    print('Making predictions with {}...'.format(cfg.model.name))
    allAblations = ["pure", "blurred5by5", "blurred3by3", "blurred7by7", "blurred_contrast", "higher_contrast", "lower_contrast"]
    #ablations = ["higher_contrast125", "lower_contrast075"]
    ablations = ["warmth90", "warmth95", "warmth105", "warmth110"]
    

    # for each ablation do predictions and wirte the images to a different \dataset\ablation folder and store the results in a pickle file with its ablation name
    for ablation in ablations:
        failedImages = []
        for i, p in enumerate(img_paths):
            if i == 1:
                ti = time()
            img = cv2.imread(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if ablation == "blurred3by3":
                imgCopy = img.copy()
                transformimg = distort.low_pass_filter(imgCopy, kernel_size=3)
                img = transformimg
            elif ablation == "blurred5by5":
                imgCopy = img.copy()
                transformimg = distort.low_pass_filter(imgCopy, kernel_size=5)
                img = transformimg
            elif ablation == "blurred7by7":
                imgCopy = img.copy()
                transformimg = distort.low_pass_filter(imgCopy, kernel_size=7)
                img = transformimg
            elif ablation == "blurred_contrast":
                imgCopy = img.copy()
                transformimg = distort.change_contrast(imgCopy)
                img = distort.low_pass_filter(transformimg)
            elif ablation == "higher_contrast125":
                imgCopy = img.copy()
                transformimg = distort.change_contrast(imgCopy, 1.25)
                img = transformimg
            elif ablation == "lower_contrast075":
                imgCopy = img.copy()
                transformimg = distort.change_contrast(imgCopy, 0.75)
                img = transformimg
            elif ablation == "warmth75":
                imgCopy = img.copy()
                transformimg = distort.change_color_warmth(imgCopy, 0.75)
                img = transformimg
            elif ablation == "warmth85":
                imgCopy = img.copy()
                transformimg = distort.change_color_warmth(imgCopy, 0.85)
                img = transformimg
            elif ablation == "warmth115":
                imgCopy = img.copy()
                transformimg = distort.change_color_warmth(imgCopy, 1.15)
                img = transformimg
            elif ablation == "warmth125":
                imgCopy = img.copy()
                transformimg = distort.change_color_warmth(imgCopy, 1.25)
                img = transformimg
            elif ablation == "warmth90":
                imgCopy = img.copy()
                transformimg = distort.change_color_warmth(imgCopy, 0.90)
                img = transformimg
            elif ablation == "warmth95":
                imgCopy = img.copy()
                transformimg = distort.change_color_warmth(imgCopy, 0.95)
                img = transformimg
            elif ablation == "warmth105":
                imgCopy = img.copy()
                transformimg = distort.change_color_warmth(imgCopy, 1.05)
                img = transformimg
            elif ablation == "warmth110":
                imgCopy = img.copy()
                transformimg = distort.change_color_warmth(imgCopy, 1.10)
                img = transformimg
            if i%10 == 0:
                print('Processing ', ablation, ' nr: ', i, 'of', len(img_paths))

            bboxes = yolo.predict(img)
            preds[i] = bboxes_to_xy(bboxes, max_darts)

            if write:
                write_dir = osp.join('./models', cfg.model.name, 'preds', split, ablation)
                #print("write dir: ", write_dir)
                os.makedirs(write_dir, exist_ok=True)
                xy = preds[i]
                xy = xy[xy[:, -1] == 1]
                error = sum(get_dart_scores(preds[i, :, :2], cfg, numeric=True)) - sum(get_dart_scores(xys[i, :, :2], cfg, numeric=True))
                if not args.fail_cases or (args.fail_cases and error != 0):
                    img = draw(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), xy[:, :2], cfg, circles=False, score=True)
                    pathToimgdir = os.path.split(p)[0]
                    imgName = os.path.split(p)[1]
                    output_path = osp.join(write_dir, os.path.split(pathToimgdir)[1], imgName)
                    os.makedirs(os.path.split(output_path)[0], exist_ok=True)
                    cv2.imwrite(output_path, img)
                    if error!= 0:
                        failedImages.append(os.path.split(output_path)[1])

        fps = (len(img_paths) - 1) / (time() - ti)
        print('FPS: {:.2f}'.format(fps))

        ASE = []  # absolute score error
        for pred, gt in zip(preds, xys):
            ASE.append(abs(
                sum(get_dart_scores(pred[:, :2], cfg, numeric=True)) -
                sum(get_dart_scores(gt[:, :2], cfg, numeric=True))))

        ASE = np.array(ASE)
        PCS = len(ASE[ASE == 0]) / len(ASE) * 100
        MASE = np.mean(ASE)

        results = {
            'failed images': failedImages,
            'fps': fps,
            'ASE': ASE.tolist() if isinstance(ASE, np.ndarray) else ASE,  # Convert numpy array to list
            'PCS': float(PCS),
            'MASE': float(MASE)
        }

        # Save results as JSON
        with open(osp.join('./models', cfg.model.name, ablation + '_results.txt'), 'w') as f:
            json.dump(results, f)


if __name__ == '__main__':
    from train import build_model
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', default='deepdarts_d1')
    parser.add_argument('-s', '--split', default='val')
    parser.add_argument('-w', '--write', action='store_true')
    parser.add_argument('-f', '--fail-cases', action='store_true')
    args = parser.parse_args()

    cfg = CN(new_allowed=True)
    cfg.merge_from_file(osp.join('configs', args.cfg + '.yaml'))
    cfg.model.name = args.cfg

    yolo = build_model(cfg)
    yolo.load_weights(osp.join('models', args.cfg, 'weights'), cfg.model.weights_type)

    predict(yolo, cfg,
            dataset=cfg.data.dataset,
            split=args.split,
            write=args.write)