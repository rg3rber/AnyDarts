import argparse
from yacs.config import CfgNode as CN
import os.path as osp
import os
from dataloaderTest import get_splits
import cv2
import numpy as np
from time import time
from dataset.annotate import draw, get_dart_scores
import pickle


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
        img_folder="myTest_PXL_bbox_crop",
        labels_path='./dataset/labels.pkl',
        dataset='myTest',
        split='test',
        max_darts=3,
        write=False):

    np.random.seed(0)

    write_dir = osp.join('./models', cfg.model.name, 'preds', split)
    if write:
        os.makedirs(write_dir, exist_ok=True)

    #data = get_splits(labels_path, dataset, split)  
    img_prefix = osp.join(cfg.data.path, 'cropped_images', str(cfg.model.input_size))
    img_paths = [osp.join(img_prefix, img_folder, name) for name in os.listdir(osp.join(img_prefix, img_folder))]

    
    """
    xys = np.zeros((len(data), 7, 3))  # third column for visibility
    data.xy = data.xy.apply(np.array)
    for i, _xy in enumerate(data.xy):
        xys[i, :_xy.shape[0], :2] = _xy
        xys[i, :_xy.shape[0], 2] = 1
    xys = xys.astype(np.float32)
    """

    preds = np.zeros((len(img_paths), 4 + max_darts, 3))
    ti = time()-0.0001
    print('Making predictions with {}...'.format(cfg.model.name))
    
    for i in range(len(img_paths)):
        img = cv2.imread(img_paths[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if i%100 == 0:
            print('Processing', i+1, 'of', len(img_paths))

        bboxes = yolo.predict(img)
        preds[i] = bboxes_to_xy(bboxes, max_darts)

        if write:
            print("writing: " + img_paths[i])
            write_dir = osp.join('.\models', cfg.model.name, 'preds', split)
            os.makedirs(write_dir, exist_ok=True)
            xy = preds[i]
            xy = xy[xy[:, -1] == 1]
            annotated_img = draw(cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR), xy[:, :2], cfg, circles=False, score=True)
            print("writing to: " + osp.join(write_dir, os.path.basename(img_paths[i])))
            cv2.imwrite(osp.join(write_dir, os.path.basename(img_paths[i])), annotated_img)

    fps = len(img_paths) / (time() - ti)
    print('FPS: {:.2f}'.format(fps))

    """
    ASE = []  # absolute score error
    for pred, gt in zip(preds, xys):
        ASE.append(abs(
            sum(get_dart_scores(pred[:, :2], cfg, numeric=True)) -
            sum(get_dart_scores(gt[:, :2], cfg, numeric=True))))

    ASE = np.array(ASE)
    PCS = len(ASE[ASE == 0]) / len(ASE) * 100
    MASE = np.mean(ASE)

    print('Percent Correct Score (PCS): {:.1f}%'.format(PCS))
    print('Mean Absolute Score Error (MASE): {:.2f}'.format(MASE))

    """

    results = {
        'img_paths': img_paths,
        'preds': preds,
        'fps': fps,
    }

    pickle.dump(results, open(osp.join('./models', cfg.model.name, 'results.pkl'), 'wb'))
    print('Saved results.')


if __name__ == '__main__':
    from train import build_model
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', default='deepdarts_d2')
    parser.add_argument('-s', '--split', default='val')
    parser.add_argument('-w', '--write', action='store_true')
    parser.add_argument('-f', '--fail-cases', action='store_true')
    parser.add_argument('-i', '--image_folder', default='myTest_PXL')
    args = parser.parse_args()

    cfg = CN(new_allowed=True)
    cfg.merge_from_file(osp.join('configs', args.cfg + '.yaml'))
    cfg.model.name = args.cfg

    yolo = build_model(cfg)
    yolo.load_weights(osp.join('models', args.cfg, 'weights'), cfg.model.weights_type)
    print('Predicting with', cfg.model.name)
    predict(yolo, cfg, img_folder=args.image_folder,
            split=args.split,
            write=args.write)