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
