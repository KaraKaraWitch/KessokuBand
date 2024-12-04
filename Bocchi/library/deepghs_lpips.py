# Pulled from waifuc. Licensed under MIT.

from typing import Dict, Literal

import numpy as np
from imgutils.metrics import lpips_difference, lpips_extract_feature
from PIL import Image


class FeatureBucket:
    def __init__(self, threshold: float = 0.45, capacity: int = 500, rtol=1.e-5, atol=1.e-8):
        self.threshold = threshold
        self.rtol, self.atol = rtol, atol
        self.features = []
        self.ratios = np.array([], dtype=float)
        self.capacity = capacity

    def check_duplicate(self, feat, ratio: float):
        for id_ in np.where(np.isclose(self.ratios, ratio, rtol=self.rtol, atol=self.atol))[0]:
            exist_feat = self.features[id_.item()]
            if lpips_difference(exist_feat, feat) <= self.threshold:
                return True

        return False

    def add(self, feat, ratio: float):
        self.features.append(feat)
        self.ratios = np.append(self.ratios, ratio)
        if len(self.features) >= self.capacity * 2:
            self.features = self.features[-self.capacity:]
            self.ratios = self.ratios[-self.capacity:]


FilterSimilarModeTyping = Literal['all', 'group']


class LPIPSFilterClass():
    def __init__(self, mode: FilterSimilarModeTyping = 'all', threshold: float = 0.45,
                 capacity: int = 500, rtol=5.e-2, atol=2.e-2):
        self.mode = mode
        self.threshold, self.rtol, self.atol = threshold, rtol, atol
        self.capacity = capacity
        self.buckets: Dict[str, FeatureBucket] = {}
        self.global_bucket = FeatureBucket(threshold, self.capacity, rtol, atol)


    def iter(self, image:Image.Image):
        ratio = image.height * 1.0 / image.width
        feat = lpips_extract_feature(image)
        bucket = self.global_bucket

        if not bucket.check_duplicate(feat, ratio):
            bucket.add(feat, ratio)
            return image

    def reset(self):
        self.buckets.clear()
        self.global_bucket = FeatureBucket(self.threshold, self.capacity, self.rtol, self.atol)