import numpy as np


def yolo_to_xyxy(bboxes, img_size):
    w, h = img_size
    
    bboxes = bboxes[:, 0:]
    bboxes[:, 0] = bboxes[:, 0]*w
    bboxes[:, 1] = bboxes[:, 1]*h
    bboxes[:, 2] = bboxes[:, 2]*w
    bboxes[:, 3] = bboxes[:, 3]*h

    bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2]/2
    bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3]/2
    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
    
    return bboxes.astype(int)

def generalize_output_format(bboxes, confidence_list, conf_thres):
    """_summary_

    Args:
        bboxes : Bounding boxes in xyxy format
        confidence_list : List containing confidence score of each class
        conf_thres : confidence_threshold

    Returns:
        np.array: Array of format [Xmin, Ymin, Xmax, Ymax, confidence, class_id]
    """
    
    class_ids = np.argmax(confidence_list, axis=1)
    conf_scr = []
    output = []
    for i, confidence in enumerate(confidence_list):
        if conf_thres < confidence[class_ids[i]]:
            conf_scr = confidence[class_ids[i]]
            res = np.append(np.append(bboxes[i], conf_scr), class_ids[i])
            output.append(res)
    return np.array(output)

def scale_bboxes(bboxes, org_img_shape, resized_img_shape):
    # Rescaling Bounding Boxes 
    bboxes[:, :4] /= np.array([resized_img_shape[1], resized_img_shape[0], resized_img_shape[1], resized_img_shape[0]])
    bboxes[:, :4] *= np.array([org_img_shape[1], org_img_shape[0], org_img_shape[1], org_img_shape[0]])  

    return bboxes