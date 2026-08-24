#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class Hand_Gestures(object):
    def __init__(
        self,
        model_path='model/Hand_Gestures/Hand_Gestures.keras',
    ):
        self.model = tf.keras.models.load_model(model_path)

    def __call__(
        self,
        landmark_list,
    ):
        # Predict expects a batch, so we expand dims
        result = self.model.predict(
            np.array([landmark_list], dtype=np.float32), verbose=0
        )
        result_index = np.argmax(np.squeeze(result))
        return result_index


# Backwards-compatible alias
KeyPointClassifier = Hand_Gestures
