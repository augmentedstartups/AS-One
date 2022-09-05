import numpy as np
import cv2

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
              'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
              'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
              'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
              'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
              'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
              'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
              'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
              'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Create a list of colors for each class where each color is a tuple of 3 integer values
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def process_output(output,  img_height, img_width,    
                   input_width, input_height,
                   conf_thres, iou_thres):
    predictions = np.squeeze(output)

    # Filter out object confidence scores below threshold
    obj_conf = predictions[:, 4]
    predictions = predictions[obj_conf > conf_thres]
    obj_conf = obj_conf[obj_conf > conf_thres]

    # Multiply class confidence with bounding box confidence
    predictions[:, 5:] *= obj_conf[:, np.newaxis]

    # Get the scores
    scores = np.max(predictions[:, 5:], axis=1)

    # Filter out the objects with a low score
    predictions = predictions[obj_conf > conf_thres]
    scores = scores[scores > conf_thres]

    # Get the class with the highest confidence
    class_ids = np.argmax(predictions[:, 5:], axis=1)

    # Get bounding boxes for each object
    boxes = extract_boxes(predictions, img_height, img_width,    
                               input_width, input_height)

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    indices = nms(boxes, scores, iou_thres)

    return boxes[indices], scores[indices], class_ids[indices]

def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def prepare_input(image, input_width, input_height):
  
    input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize input image
    input_img = cv2.resize(input_img, (input_width, input_height))

    # Scale input pixel values to 0 to 1
    input_img = input_img / 255.0
    input_img = input_img.transpose(2, 0, 1)
    input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

    return input_tensor

def extract_boxes(predictions, img_height, img_width,    
                 input_width, input_height):
    # Extract boxes from predictions
    boxes = predictions[:, :4]

    # Scale boxes to original image dimensions
    boxes /= np.array([input_width, input_height, input_width, input_height])
    boxes *= np.array([img_width, img_height, img_width, img_height])

    # Convert boxes to xyxy format
    boxes = xywh2xyxy(boxes)

    return boxes