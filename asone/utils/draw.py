import cv2
from numpy import random
import numpy as np
from asone.utils import compute_color_for_labels
from asone.utils import get_names
from collections import deque

names = get_names()
data_deque = {}


def draw_ui_box(x, img, label=None, color=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(str(label), 0, fontScale=tl / 3, thickness=tf)[0]
        # c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        img = draw_border(img, (c1[0], c1[1] - t_size[1] - 3),
                          (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)

        # cv2.line(img, c1, c2, color, 30)
        # cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, str(label), (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2
    # Top leftfrom collections import deque (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)

    cv2.circle(img, (x1 + r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 - r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 + r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 - r, y2-r), 2, color, 12)

    return img


def draw_boxes(img, bbox_xyxy, class_ids, identities=None, draw_trails=False, offset=(0, 0), class_names=None):
    # cv2.line(img, line2[0], line2[1], (0,200,0), 3)
    height, width, _ = img.shape

    # remove tracked point from buffer if object is lost
    if draw_trails:
        for key in list(data_deque):
            if key not in identities:
                data_deque.pop(key)
    
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # get ID of object
        id = int(identities[i]) if identities is not None else None
        
        # if class_ids is not None:
        color = compute_color_for_labels(int(class_ids[i]))

        if class_names:
            obj_name = class_names[int(class_ids[i])]
        else:                
            obj_name = names[int(class_ids[i])]
        
        label = f'{obj_name}' if id is None else f'{id}'

        draw_ui_box(box, img, label=label, color=color, line_thickness=2)

        # Draw trails        
        # code to find center of bottom edge
        center = (int((x2+x1) / 2), int((y2+y2)/2))

        if draw_trails:
            # create new buffer for new object
            if id not in data_deque:  
                data_deque[id] = deque(maxlen= 64)
        
            data_deque[id].appendleft(center)    
            drawtrails(data_deque, id, color, img)
 
    return img

def drawtrails(data_deque, id, color, img):
    # draw trail
    for i in range(1, len(data_deque[id])):
        # check if on buffer value is none
        if data_deque[id][i - 1] is None or data_deque[id][i] is None:
            continue

        # generate dynamic thickness of trails
        thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
        
        # draw trails
        cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)

def draw_text(img, results, offset=(0, 0)):
    color = compute_color_for_labels(int(0))
    for res in results:
        box = res[:4]
        text = res[4]
        x1, y1, x2, y2 = box
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        label = text

        draw_ui_box(box, img, label=label, color=color, line_thickness=2)
        center = (int((x2+x1) / 2), int((y2+y2)/2))
 
    return img

# Utils for code estimation 

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                                      [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                                      [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                                      [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],
                                     dtype=np.uint8)

    def __call__(self, i, bgr=False):
        """Converts hex color codes to rgb values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


def draw_kpts(image ,keypoints, shape=(640, 640), radius=5, kpt_line=True):
        """Plot keypoints on the image.
        Args:
            kpts (tensor): Predicted keypoints with shape [17, 3]. Each keypoint has (x, y, confidence).
            shape (tuple): Image shape as a tuple (h, w), where h is the height and w is the width.
            radius (int, optional): Radius of the drawn keypoints. Default is 5.
            kpt_line (bool, optional): If True, the function will draw lines connecting keypoints
                                       for human pose. Default is True.
        Note: `kpt_line=True` currently only supports human pose plotting.
        """
        # if self.pil:
        #     # Convert to numpy first
        #     self.im = np.asarray(self.im).copy()
        if keypoints is not None:
            for kpts in reversed(keypoints):
                
                limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
                kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
                skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                                [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
                nkpt, ndim = kpts.shape
                is_pose = nkpt == 17 and ndim == 3
                kpt_line &= is_pose  # `kpt_line=True` for now only supports human pose plotting
                for i, k in enumerate(kpts):
                    color_k = [int(x) for x in kpt_color[i]] if is_pose else colors(i)
                    x_coord, y_coord = k[0], k[1]
                    if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                        if len(k) == 3:
                            conf = k[2]
                            if conf < 0.5:
                                continue
                        cv2.circle(image, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)

                if kpt_line:
                    ndim = kpts.shape[-1]
                    for i, sk in enumerate(skeleton):
                        pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
                        pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
                        if ndim == 3:
                            conf1 = kpts[(sk[0] - 1), 2]
                            conf2 = kpts[(sk[1] - 1), 2]
                            if conf1 < 0.5 or conf2 < 0.5:
                                continue
                        if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                            continue
                        if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                            continue
                        cv2.line(image, pos1, pos2, [int(x) for x in limb_color[i]], thickness=2, lineType=cv2.LINE_AA)
                
        return image

def plot_skeleton_kpts(im, kpts, steps, orig_shape=None):
    #Plot the skeleton and keypointsfor coco datatset
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 5
    num_kpts = len(kpts) // steps

    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2]
                if conf < 0.5:
                    continue
            overlay = im.copy()
            alpha = 0.4
            cv2.circle(overlay, (int(x_coord), int(y_coord)), 8, (int(220), int(237), int(245)), 8)
            cv2.circle(im, (int(x_coord), int(y_coord)), 5, (int(255), int(255), int(255)), -1)
            # im = output
            cv2.addWeighted(overlay, alpha, im, 1 - alpha, 0, im)

    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
        pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
        if steps == 3:
            conf1 = kpts[(sk[0]-1)*steps+2]
            conf2 = kpts[(sk[1]-1)*steps+2]
            if conf1<0.5 or conf2<0.5:
                continue
        if pos1[0]%640 == 0 or pos1[1]%640==0 or pos1[0]<0 or pos1[1]<0:
            continue
        if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0]<0 or pos2[1]<0:
            continue
        cv2.line(im, pos1, pos2, (int(255), int(255), int(255)), thickness=2)
