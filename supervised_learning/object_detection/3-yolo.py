#!/usr/bin/env python3
"""Module that performs non-max suppression on filtered boxes."""
import numpy as np
from tensorflow import keras as K


class Yolo:
    """Uses the Yolo v3 algorithm to perform object detection."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initializes the Yolo class.

        Args:
            model_path: path to where a Darknet Keras model is stored.
            classes_path: path to the list of class names for the model.
            class_t: float box score threshold for the initial filtering.
            nms_t: float IOU threshold for non-max suppression.
            anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2).
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Processes the outputs of the Darknet model.

        Args:
            outputs: list of numpy.ndarrays containing predictions.
            image_size: numpy.ndarray with the image's original size.

        Returns:
            Tuple of (boxes, box_confidences, box_class_probs).
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_h, image_w = image_size[0], image_size[1]
        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            t_xy = output[..., 0:2]
            t_wh = output[..., 2:4]

            box_conf = 1 / (1 + np.exp(-output[..., 4:5]))
            box_class = 1 / (1 + np.exp(-output[..., 5:]))
            box_confidences.append(box_conf)
            box_class_probs.append(box_class)

            sig_xy = 1 / (1 + np.exp(-t_xy))

            cx = np.arange(grid_w).reshape(1, grid_w, 1)
            cy = np.arange(grid_h).reshape(grid_h, 1, 1)
            cx = np.tile(cx, (grid_h, 1, anchor_boxes))
            cy = np.tile(cy, (1, grid_w, anchor_boxes))

            bx = (sig_xy[..., 0] + cx) / grid_w
            by = (sig_xy[..., 1] + cy) / grid_h

            anchors_i = self.anchors[i]
            pw = anchors_i[:, 0]
            ph = anchors_i[:, 1]

            bw = (np.exp(t_wh[..., 0]) * pw) / input_w
            bh = (np.exp(t_wh[..., 1]) * ph) / input_h

            x1 = (bx - bw / 2) * image_w
            y1 = (by - bh / 2) * image_h
            x2 = (bx + bw / 2) * image_w
            y2 = (by + bh / 2) * image_h

            box = np.zeros(output[..., 0:4].shape)
            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2
            boxes.append(box)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filters boxes based on class score threshold.

        Args:
            boxes: list of numpy.ndarrays containing processed boxes.
            box_confidences: list of numpy.ndarrays with box confidences.
            box_class_probs: list of numpy.ndarrays with class probabilities.

        Returns:
            Tuple of (filtered_boxes, box_classes, box_scores).
        """
        all_boxes = []
        all_classes = []
        all_scores = []

        for i in range(len(boxes)):
            scores = box_confidences[i] * box_class_probs[i]
            classes = np.argmax(scores, axis=-1)
            class_scores = np.max(scores, axis=-1)

            mask = class_scores >= self.class_t

            all_boxes.append(boxes[i][mask])
            all_classes.append(classes[mask])
            all_scores.append(class_scores[mask])

        filtered_boxes = np.concatenate(all_boxes, axis=0)
        box_classes = np.concatenate(all_classes, axis=0)
        box_scores = np.concatenate(all_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Performs non-max suppression on filtered boxes.

        Args:
            filtered_boxes: numpy.ndarray of shape (?, 4) with boxes.
            box_classes: numpy.ndarray of shape (?,) with class numbers.
            box_scores: numpy.ndarray of shape (?,) with box scores.

        Returns:
            Tuple of (box_predictions, predicted_box_classes,
            predicted_box_scores).
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            idx = np.where(box_classes == cls)[0]
            cls_boxes = filtered_boxes[idx]
            cls_scores = box_scores[idx]

            order = cls_scores.argsort()[::-1]
            cls_boxes = cls_boxes[order]
            cls_scores = cls_scores[order]

            keep = []
            while len(cls_boxes) > 0:
                keep.append(0)
                if len(cls_boxes) == 1:
                    break

                x1 = np.maximum(cls_boxes[0, 0], cls_boxes[1:, 0])
                y1 = np.maximum(cls_boxes[0, 1], cls_boxes[1:, 1])
                x2 = np.minimum(cls_boxes[0, 2], cls_boxes[1:, 2])
                y2 = np.minimum(cls_boxes[0, 3], cls_boxes[1:, 3])

                w = np.maximum(0, x2 - x1)
                h = np.maximum(0, y2 - y1)
                inter = w * h

                area0 = ((cls_boxes[0, 2] - cls_boxes[0, 0]) *
                         (cls_boxes[0, 3] - cls_boxes[0, 1]))
                areas = ((cls_boxes[1:, 2] - cls_boxes[1:, 0]) *
                         (cls_boxes[1:, 3] - cls_boxes[1:, 1]))
                union = area0 + areas - inter
                iou = inter / union

                mask = iou < self.nms_t
                box_predictions.append(cls_boxes[0])
                predicted_box_classes.append(cls)
                predicted_box_scores.append(cls_scores[0])

                cls_boxes = cls_boxes[1:][mask]
                cls_scores = cls_scores[1:][mask]

        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores
