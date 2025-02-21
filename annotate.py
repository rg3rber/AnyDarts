import os
import os.path as osp
import cv2
import pandas as pd
import numpy as np
from yacs.config import CfgNode as CN
import argparse
from BoardDetection import find_board
from HelperFunctions import get_dart_scores, get_circle, board_radii, draw, draw_circles

def crop_board(img_path, bbox=None, crop_info=(0, 0, 0), crop_pad=1.1):
    img = cv2.imread(img_path)
    if bbox is None:
        x, y, r = crop_info
        r = int(r * crop_pad)
        bbox = [y-r, y+r, x-r, x+r]
    crop = img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    return crop, bbox


def on_click(event, x, y, flags, param):
    global xy, img_copy
    h, w = img_copy.shape[:2]
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(xy) < 7:
            xy.append([x/w, y/h])
            print_xy()
        else:
            print('Already annotated 7 points.')


def print_xy():
    global xy
    names = {
        0: 'cal_1', 1: 'cal_2', 2: 'cal_3', 3: 'cal_4',
        4: 'dart_1', 5: 'dart_2', 6: 'dart_3'}
    print('{}: {}'.format(names[len(xy)-1], xy[-1]))


def get_ellipses(xy, r_double=0.17, r_treble=0.1074):
    c = np.mean(xy[:4], axis=0)
    a1_double = ((xy[2][0] - xy[3][0]) ** 2 + (xy[2][1] - xy[3][1]) ** 2) ** 0.5 / 2
    a2_double = ((xy[0][0] - xy[1][0]) ** 2 + (xy[0][1] - xy[1][1]) ** 2) ** 0.5 / 2
    a1_treble = a1_double * (r_treble / r_double)
    a2_treble = a2_double * (r_treble / r_double)
    angle = np.arctan((xy[3, 1] - c[1]) / (xy[3, 0] - c[0])) / np.pi * 180
    return c, [a1_double, a2_double], [a1_treble, a2_treble], angle


def draw_ellipses(img, xy, num_pts=7):
    # img must be uint8
    xy = np.array(xy)
    if xy.shape[0] > num_pts:
        xy = xy.reshape((-1, 2))
    if np.mean(xy) < 1:
        h, w = img.shape[:2]
        xy[:, 0] *= w
        xy[:, 1] *= h
    c, a_double, a_treble, angle = get_ellipses(xy)
    angle = np.arctan((xy[3,1]-c[1])/(xy[3,0]-c[0]))/np.pi*180
    cv2.ellipse(img, (int(round(c[0])), int(round(c[1]))),
                (int(round(a_double[0])), int(round(a_double[1]))),
                int(round(angle)), 0, 360, (255, 255, 255))
    cv2.ellipse(img, (int(round(c[0])), int(round(c[1]))),
                (int(round(a_treble[0])), int(round(a_treble[1]))),
                int(round(angle)), 0, 360, (255, 255, 255))
    return img

def adjust_xy(idx):
    global xy, img_copy
    key = cv2.waitKey(0) & 0xFF
    xy = np.array(xy)
    h, w = img_copy.shape[:2]
    xy[:, 0] *= w; xy[:, 1] *= h
    if key == 52:  # one pixel left
        if idx == -1:
            xy[:, 0] -= 1
        else:
            xy[idx, 0] -= 1
    if key == 56:  # one pixel up
        if idx == -1:
            xy[:, 1] -= 1
        else:
            xy[idx, 1] -= 1
    if key == 54:  # one pixel right
        if idx == -1:
            xy[:, 0] += 1
        else:
            xy[idx, 0] += 1
    if key == 50:  # one pixel down
        if idx == -1:
            xy[:, 1] += 1
        else:
            xy[idx, 1] += 1
    xy[:, 0] /= w; xy[:, 1] /= h
    xy = xy.tolist()


def add_last_dart(annot, data_path, folder):
    csv_path = osp.join(data_path, 'annotations', folder + '.csv')
    if osp.isfile(csv_path):
        dart_labels = []
        csv = pd.read_csv(csv_path)
        for idx in csv.index.values:
            for c in csv.columns:
                dart_labels.append(str(csv.loc[idx, c]))
        annot['last_dart'] = dart_labels
    return annot

def get_bounding_box(img_path, scale=0.2):
    bbox = None
    try:
        bbox = find_board(img_path)
    except Exception:
        print('Error finding board, switch to manual annotation')
        bbox = get_bounding_box_old(img_path, scale)
    return bbox

def get_bounding_box_old(img_path, scale=0.5):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, None, fx=scale, fy=scale)
    h, w = img_resized.shape[:2]
    xy_bbox = []

    def on_click_bbox(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(xy_bbox) < 2:
                xy_bbox.append([
                    round((x / w) * img.shape[1]),
                    round((y / h) * img.shape[0])])

    window = 'get bbox'
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_click_bbox)
    while len(xy_bbox) < 2:
        # print(xy_bbox)
        cv2.imshow(window, img_resized)
        key = cv2.waitKey(100)
        if key == ord('q'):  # quit
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()
    assert len(xy_bbox) == 2, 'click 2 points to get bounding box'
    xy_bbox = np.array(xy_bbox)
    # bbox = [y1 y2 x1 x2]
    bbox = [min(xy_bbox[:, 1]), max(xy_bbox[:, 1]), min(xy_bbox[:, 0]), max(xy_bbox[:, 0])]
    print('Bounding box:', bbox)
    return bbox


def main(cfg, folder, scale, draw_circles, dart_score=True):
    global xy, img_copy
    img_dir = osp.join('dataset/images', folder)
    imgs = sorted(os.listdir(img_dir))
    annot_path = osp.join('dataset/annotations', folder + '.pkl')
    if osp.isfile(annot_path):
        annot = pd.read_pickle(annot_path)
    else:
        annot = pd.DataFrame(columns=['img_name', 'bbox', 'xy'])
        annot['img_name'] = imgs
        annot['bbox'] = None
        annot['xy'] = None
        annot = add_last_dart(annot, 'dataset/', folder)

    i = 0
    for j in range(len(annot)):
        a = annot.iloc[j,:]
        if a['bbox'] is not None:
            i = j

    while i < len(imgs):
        xy = []
        a = annot.iloc[i,:]
        print('Annotating {}'.format(a['img_name']))
        if a['bbox'] is None:
            if i == 0:
                bbox = get_bounding_box(osp.join(img_dir, a['img_name']))
            if i > 0:
                last_a = annot.iloc[i-1,:]
                if last_a['xy'] is not None:
                    xy = last_a['xy'].copy()
            else:
                xy = []
        elif a['xy'] is None:
            bbox = a['bbox']
        else:
            bbox, xy = a['bbox'], a['xy']

        crop, _ = crop_board(osp.join(img_dir, a['img_name']), bbox=bbox)
        crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)))
        cv2.putText(crop, '{}/{} {}'.format(i+1, len(annot), a['img_name']), (0, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        img_copy = crop.copy()

        cv2.namedWindow(folder)
        cv2.setMouseCallback(folder, on_click)
        while True:
            img_copy = draw(img_copy, np.array(xy), cfg, draw_circles, dart_score)
            cv2.imshow(folder, img_copy)
            key = cv2.waitKey(100) & 0xFF  # update every 100 ms

            if key == ord('q'):  # quit
                cv2.destroyAllWindows()
                i = len(imgs)
                break

            if key == ord('b'):  # draw new bounding box
                idx = annot[(annot['img_name'] == a['img_name'])].index.values[0]
                annot.at[idx, 'bbox'] = get_bounding_box(osp.join(img_dir, a['img_name']), scale)
                break

            if key == ord('m'):  # draw manual bounding box
                idx = annot[(annot['img_name'] == a['img_name'])].index.values[0]
                annot.at[idx, 'bbox'] = get_bounding_box_old(osp.join(img_dir, a['img_name']), scale)
                break

            if key == ord('.'):
                i += 1
                img_copy = crop.copy()
                break

            if key == ord(','):
                if i > 0:
                    i += -1
                    img_copy = crop.copy()
                    break

            if key == ord('z'):  # undo keypoint
                xy = xy[:-1]
                img_copy = crop.copy()

            if key == ord('x'):  # reset annotation
                idx = annot[(annot['img_name'] == a['img_name'])].index.values[0]
                annot.at[idx, 'xy'] = None
                annot.to_pickle(annot_path)
                break

            if key == ord('d'):  # delete img
                print('Are you sure you want to delete this image? (y/n)')
                key = cv2.waitKey(0) & 0xFF
                if key == ord('y'):
                    idx = annot[(annot['img_name'] == a['img_name'])].index.values[0]
                    annot = annot.drop([idx])
                    annot.to_pickle(annot_path)
                    os.remove(osp.join(img_dir, a['img_name']))
                    print('Deleted image {}'.format(a['img_name']))
                    break
                else:
                    print('Image not deleted.')
                    continue

            if key == ord('a'):  # accept keypoints
                idx = annot[(annot['img_name'] == a['img_name'])].index.values[0]
                annot.at[idx, 'xy'] = xy
                annot.at[idx, 'bbox'] = bbox
                annot.to_pickle(annot_path)
                i += 1
                break

            if key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('0')]:
                adjust_xy(idx=key - 49)  # ord('1') = 49
                img_copy = crop.copy()
                continue


if __name__ == '__main__':
    import sys
    sys.path.append('../../')
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--img-folder', default='d3_25_12_2024_easy1')
    parser.add_argument('-s', '--scale', type=float, default=0.5)
    parser.add_argument('-d', '--draw-circles', action='store_true')
    args = parser.parse_args()

    cfg = CN(new_allowed=True)
    cfg.merge_from_file('configs/holo_v2.yaml')

    main(cfg, args.img_folder, args.scale, args.draw_circles)
