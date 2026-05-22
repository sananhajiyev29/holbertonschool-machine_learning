#!/usr/bin/env python3
"""Module that initializes the Yolo class for object detection."""
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
            anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
                containing all of the anchor boxes.
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
