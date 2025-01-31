import os.path as osp
import os
import cv2
import numpy as np

""" converts array of bounding boxes [x_center, y_center, width, height, class] to xy [x, y, visible]
 with index = class label"""
def bboxes_to_xy(bboxes, max_darts=3):
    xy = np.zeros((4+max_darts, 3), dtype=np.float32)
    hadToEstimate = []
    dart_index = 0
    for box in bboxes: # iterate over boxes in decreasing order of confidence
        if int(box[4]) == 4 and dart_index < max_darts: # if box is a dart and we have not reached max_darts
            dart_xys = box[:2]
            xy[4 + dart_index, :2] = dart_xys
            dart_index += 1
        else:
            if np.all(xy[int(box[4]), :3] == 0): # if there is no entry for this class yet
                xy[int(box[4]), :2] = box[:2]

    xy[(xy[:, 0] > 0) & (xy[:, 1] > 0), -1] = 1 # set valid to 1 if x and y are visible (>0)

    if np.sum(xy[:4, -1]) == 4: # if all calibration points are visible
        return xy, hadToEstimate
    else:
        hadToEstimate = np.where(xy[:4, -1] == 0)[0] # hold all indices where the last column was 0
        hadToEstimate = [i + 1 for i in hadToEstimate] # increment each element by 1 to get the corresponding cal point
        xy = est_cal_pts(xy) # estimate invisible calibration points
    return xy, hadToEstimate

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

def get_circle(xy):
    c = np.mean(xy[:4], axis=0)
    if np.isnan(c).any():  # Check for NaN values
        return None, None
    r = np.mean(np.linalg.norm(xy[:4] - c, axis=-1))
    if np.isnan(r):  # Check for NaN radius
        return None, None
    return c, r


def board_radii(r_d, cfg):
    r_t = r_d * (cfg.board.r_treble / cfg.board.r_double)  # treble radius, in px
    r_ib = r_d * (cfg.board.r_inner_bull / cfg.board.r_double)  # inner bull radius, in px
    r_ob = r_d * (cfg.board.r_outer_bull / cfg.board.r_double) # outer bull radius, in px
    w_dt = cfg.board.w_double_treble * (r_d / cfg.board.r_double)  # width of double and treble
    return r_t, r_ob, r_ib, w_dt


def draw_circles(img, xy, cfg, color=(255, 255, 255)):
    c, r_d = get_circle(xy)  # double radius
    if c is None or r_d is None:  # Skip drawing if invalid values
        return img
    r_t, r_ob, r_ib, w_dt = board_radii(r_d, cfg)
    center = (int(round(c[0])), int(round(c[1])))
    for r in [r_d, r_d - w_dt, r_t, r_t - w_dt, r_ib, r_ob]:
        cv2.circle(img, center, int(r), color)
    return img

def transform(xy, img=None, angle=9, M=None):
    if xy.shape[-1] == 3:
        has_vis = True
        vis = xy[:, 2:]
        xy = xy[:, :2]
    else:
        has_vis = False

    if img is not None and np.mean(xy[:4]) < 1:
        h, w = img.shape[:2]
        xy *= [[w, h]]

    if M is None:
        c, r = get_circle(xy)  # not necessarily a circle
        # c is center of 4 calibration points, r is mean distance from center to calibration points

        src_pts = xy[:4].astype(np.float32)
        dst_pts = np.array([
            [c[0] - r * np.sin(np.deg2rad(angle)), c[1] - r * np.cos(np.deg2rad(angle))],
            [c[0] + r * np.sin(np.deg2rad(angle)), c[1] + r * np.cos(np.deg2rad(angle))],
            [c[0] - r * np.cos(np.deg2rad(angle)), c[1] + r * np.sin(np.deg2rad(angle))],
            [c[0] + r * np.cos(np.deg2rad(angle)), c[1] - r * np.sin(np.deg2rad(angle))]
        ]).astype(np.float32)
        if src_pts.shape[0] == 4 and dst_pts.shape[0] == 4:
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        else:
            raise ValueError("src_pts and dst_pts must contain exactly 4 points each")

    xyz = np.concatenate((xy, np.ones((xy.shape[0], 1))), axis=-1).astype(np.float32)
    xyz_dst = np.matmul(M, xyz.T).T
    xy_dst = xyz_dst[:, :2] / xyz_dst[:, 2:]

    if img is not None:
        img = cv2.warpPerspective(img.copy(), M, (img.shape[1], img.shape[0]))
        xy_dst /= [[w, h]]

    if has_vis:
        xy_dst = np.concatenate([xy_dst, vis], axis=-1)

    return xy_dst, img, M

# get_dart_scores: returns the score of each dart throw based on the xy pixel coordinates of the darts
def get_dart_scores(xy, cfg, numeric=False):
    valid_cal_pts = xy[:4][(xy[:4, 0] > 0) & (xy[:4, 1] > 0)]
    if xy.shape[0] <= 4 or valid_cal_pts.shape[0] < 4:  # missing calibration point
        return []
    xy, _, _ = transform(xy.copy(), angle=0)
    c, r_d = get_circle(xy)
    r_t, r_ob, r_ib, w_dt = board_radii(r_d, cfg)
    xy -= c
    angles = np.arctan2(-xy[4:, 1], xy[4:, 0]) / np.pi * 180
    angles = [a + 360 if a < 0 else a for a in angles]  # map to 0-360
    distances = np.linalg.norm(xy[4:], axis=-1)
    scores = []
    for angle, dist in zip(angles, distances):
        if dist > r_d:
            scores.append('0')
        elif dist <= r_ib:
            scores.append('DB')
        elif dist <= r_ob:
            scores.append('B')
        else:
            board_dict = cfg.BOARD_DICT
            number = board_dict[int(angle / 18)]
            if dist <= r_d and dist > r_d - w_dt:
                scores.append('D' + number)
            elif dist <= r_t and dist > r_t - w_dt:
                scores.append('T' + number)
            else:
                scores.append(number)
    if numeric:
        for i, s in enumerate(scores):
            if 'B' in s:
                if 'D' in s:
                    scores[i] = 50
                else:
                    scores[i] = 25
            else:
                if 'D' in s or 'T' in s:
                    scores[i] = int(s[1:])
                    scores[i] = scores[i] * 2 if 'D' in s else scores[i] * 3
                else:
                    scores[i] = int(s)
    return scores

def draw(img, xy, cfg, circles, score, color=(255, 255, 0)):
    xy = np.array(xy)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    line_type = 5
    if xy.shape[0] > 7:
        xy = xy.reshape((-1, 2))
    if np.mean(xy) < 1:
        h, w = img.shape[:2]
        xy[:, 0] *= w
        xy[:, 1] *= h
    if xy.shape[0] >= 4 and circles:
        img = draw_circles(img, xy, cfg)
    if xy.shape[0] > 4 and score:
        scores = get_dart_scores(xy, cfg)
        cv2.putText(img, str(total_score(scores)), (50, 50), font,
                    font_scale, (255, 255, 255), line_type)
    for i, [x, y] in enumerate(xy):
        if i < 4:
            c = (0, 255, 0)  # green
        else:
            c = color  # cyan
        x = int(round(x))
        y = int(round(y))
        if i >= 4:
            cv2.circle(img, (x, y), 10, c, 1)
            cv2.circle(img, (x, y), 1, c, -1) # draw the center
            if score:
                txt = str(scores[i - 4])
            else:
                txt = str(i + 1)
            cv2.putText(img, txt, (x + 8, y), font,
                    font_scale, c, line_type)
        else:
            cv2.circle(img, (x, y), 10, c, 1)
            cv2.circle(img, (x, y), 1, c, -1) # draw the center
            cv2.putText(img, str(i + 1), (x + 8, y), font,
                        font_scale/2, c, line_type)
    return img

def total_score(scores):

    if len(scores) == 0:
        return 0

    total = 0

    for score in scores:
        try:
            total += int(score)
        except ValueError:
            if score == "DB":
                total += 50
                continue
            elif score == "B":
                total += 25

            elif score[0] == "D":
                total += int(score[1:]) * 2
            elif score[0] == "T":
                total += int(score[1:]) * 3

    return total
