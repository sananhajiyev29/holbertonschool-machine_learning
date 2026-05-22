#!/usr/bin/env python3
"""Module that processes the outputs of the Yolo model."""
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
            outputs: list of numpy.ndarrays containing predictions from
                the Darknet model for a single image.
            image_size: numpy.ndarray with the image's original size
                [image_height, image_width].

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
