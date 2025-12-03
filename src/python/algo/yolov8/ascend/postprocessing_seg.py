import math
import time
import cv2
import numpy as np


NUM_MASKS = 32


def postprocessing_seg(outputs, conf, iou, imgz, image_shape):
    boxes, scores, class_ids, mask_pred = process_box_output(outputs[0], conf, iou, imgz, image_shape)
    mask_maps = process_mask_output(mask_pred, outputs[1], boxes, image_shape)

    return boxes, class_ids, scores, mask_maps

def process_box_output(box_output, conf_threshold, iou_threshold, imgz, image_shape):
    predictions = np.squeeze(box_output).T
    num_classes = box_output.shape[1] - NUM_MASKS - 4

    # Filter out object confidence scores below threshold
    scores = np.max(predictions[:, 4:4+num_classes], axis=1)
    predictions = predictions[scores > conf_threshold, :]
    scores = scores[scores > conf_threshold]

    if len(scores) == 0:
        return [], [], [], np.array([])

    box_predictions = predictions[..., :num_classes+4]
    mask_predictions = predictions[..., num_classes+4:]

    # Get the class with the highest confidence
    class_ids = np.argmax(box_predictions[:, 4:], axis=1)

    # Get bounding boxes for each object
    boxes = extract_boxes(box_predictions, imgz, image_shape)

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    indices = nms(boxes, scores, iou_threshold)

    return boxes[indices].astype(int).tolist(), scores[indices].tolist(), class_ids[indices].tolist(), mask_predictions[indices]


def extract_boxes(box_predictions, imgz, image_shape):
    # Extract boxes from predictions
    boxes = box_predictions[:, :4]

    # Scale boxes to original image dimensions
    boxes = rescale_boxes(boxes, (imgz, imgz), (image_shape[1], image_shape[0]))

    # Convert boxes to xyxy format
    boxes = xywh2xyxy(boxes)

    # Check the boxes are within the image
    boxes[:, 0] = np.clip(boxes[:, 0], 0, image_shape[0])
    boxes[:, 1] = np.clip(boxes[:, 1], 0, image_shape[1])
    boxes[:, 2] = np.clip(boxes[:, 2], 0, image_shape[0])
    boxes[:, 3] = np.clip(boxes[:, 3], 0, image_shape[1])

    return boxes

def rescale_boxes(boxes, input_shape, image_shape):
    # Rescale boxes to original image dimensions
    input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])

    return boxes

def process_mask_output(mask_predictions, mask_output, boxes, image_shape):
    if mask_predictions.shape[0] == 0:
        return []

    mask_output = np.squeeze(mask_output)

    # Calculate the mask maps for each box
    num_mask, mask_height, mask_width = mask_output.shape  # CHW
    masks = sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
    masks = masks.reshape((-1, mask_height, mask_width))

    # Downscale the boxes to match the mask size
    scale_boxes = rescale_boxes(boxes,
                                (image_shape[1], image_shape[0]),
                                (mask_height, mask_width))

    # 初始化时直接指定为uint8类型
    mask_maps = np.zeros((len(scale_boxes), image_shape[1], image_shape[0]), dtype=np.uint8)
    blur_size = (int(image_shape[0] / mask_width), int(image_shape[1] / mask_height))
    for i in range(len(scale_boxes)):
        scale_x1 = int(math.floor(scale_boxes[i][0]))
        scale_y1 = int(math.floor(scale_boxes[i][1]))
        scale_x2 = int(math.ceil(scale_boxes[i][2]))
        scale_y2 = int(math.ceil(scale_boxes[i][3]))

        x1 = int(math.floor(boxes[i][0]))
        y1 = int(math.floor(boxes[i][1]))
        x2 = int(math.ceil(boxes[i][2]))
        y2 = int(math.ceil(boxes[i][3]))

        scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
        crop_mask = cv2.resize(scale_crop_mask,
                            (x2 - x1, y2 - y1),
                            interpolation=cv2.INTER_CUBIC)

        crop_mask = cv2.blur(crop_mask, blur_size)

        crop_mask = (crop_mask > 0.5).astype(np.uint8)
        mask_maps[i, y1:y2, x1:x2] = crop_mask

    return mask_maps

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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
