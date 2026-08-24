#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class Finger_Gestures(object):
    def __init__(
        self,
        model_path='model/Finger_Gestures/Finger_Gestures.keras',
        score_th=0.5,
        invalid_value=0,
    ):
        self.model = tf.keras.models.load_model(model_path)
        self.score_th = score_th
        self.invalid_value = invalid_value

    def __call__(
        self,
        point_history,
    ):
        result = self.model.predict(
            np.array([point_history], dtype=np.float32), verbose=0
        )
        result_squeeze = np.squeeze(result)
        result_index = np.argmax(result_squeeze)

        if result_squeeze[result_index] < self.score_th:
            result_index = self.invalid_value

        return result_index


# Backwards-compatible alias
PointHistoryClassifier = Finger_Gestures
