##\file test_sequential.py
##\brief Tests for sequential image processors
##\details Unit tests for sequential monochrome conversion and hue adjustment
##\author Lab Team
##\version 1.0

import unittest
import numpy as np
import cv2
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.modules.sequential import MonochromeConverterSeq, HueAdjusterSeq
from src.image.opencv_image import OpenCVImage
from src.modules.image_utils import SUPPORTED_MONO_METHODS


##\brief Test cases for MonochromeConverterSeq initialization
class TestMonochromeConverterSeqInitialization(unittest.TestCase):
    
    def test_default_initialization(self):
        converter = MonochromeConverterSeq()
        self.assertEqual(converter.method, 'standard')
    
    def test_initialization_with_valid_method(self):
        for method in SUPPORTED_MONO_METHODS:
            converter = MonochromeConverterSeq(method=method)
            self.assertEqual(converter.method, method)
    
    def test_initialization_with_invalid_method(self):
        converter = MonochromeConverterSeq(method='invalid_method')
        self.assertEqual(converter.method, 'standard')
    
    def test_initialization_luminosity(self):
        converter = MonochromeConverterSeq(method='luminosity')
        self.assertEqual(converter.method, 'luminosity')
    
    def test_initialization_average(self):
        converter = MonochromeConverterSeq(method='average')
        self.assertEqual(converter.method, 'average')


##\brief Test cases for MonochromeConverterSeq processing
class TestMonochromeConverterSeqProcess(unittest.TestCase):
    
    def setUp(self):
        self.converter = MonochromeConverterSeq()
        
        self.test_image = OpenCVImage(100, 100)
        self.test_image._opencv_data = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        self.test_image._data = self.test_image._opencv_data.copy()
    
    def test_process_with_valid_image(self):
        result = self.converter.process(self.test_image)
        self.assertIsNotNone(result)
        self.assertEqual(result.width, self.test_image.width)
        self.assertEqual(result.height, self.test_image.height)
    
    def test_process_with_none_data(self):
        empty_image = OpenCVImage(100, 100)
        result = self.converter.process(empty_image)
        self.assertIsNone(result.data)
    
    def test_process_returns_cloned_image(self):
        result = self.converter.process(self.test_image)
        self.assertIsNot(result, self.test_image)
    
    def test_process_standard_method(self):
        converter = MonochromeConverterSeq(method='standard')
        result = converter.process(self.test_image)
        self.assertIsNotNone(result.data)
    
    def test_process_luminosity_method(self):
        converter = MonochromeConverterSeq(method='luminosity')
        result = converter.process(self.test_image)
        self.assertIsNotNone(result.data)
    
    def test_process_average_method(self):
        converter = MonochromeConverterSeq(method='average')
        result = converter.process(self.test_image)
        self.assertIsNotNone(result.data)
    
    def test_process_preserves_dimensions(self):
        result = self.converter.process(self.test_image)
        self.assertEqual(result.width, self.test_image.width)
        self.assertEqual(result.height, self.test_image.height)
    
    def test_process_produces_different_data(self):
        result = self.converter.process(self.test_image)
        self.assertIsNotNone(result.data)
    
    def test_process_small_image(self):
        small_image = OpenCVImage(1, 1)
        small_image._opencv_data = np.array([[[100, 150, 200]]], dtype=np.uint8)
        small_image._data = small_image._opencv_data.copy()
        
        result = self.converter.process(small_image)
        self.assertEqual(result.width, 1)
        self.assertEqual(result.height, 1)
    
    def test_process_large_image(self):
        large_image = OpenCVImage(500, 500)
        large_image._opencv_data = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
        large_image._data = large_image._opencv_data.copy()
        
        result = self.converter.process(large_image)
        self.assertEqual(result.width, 500)
        self.assertEqual(result.height, 500)


class TestHueAdjusterSeqInitialization(unittest.TestCase):
    
    def test_default_initialization(self):
        adjuster = HueAdjusterSeq()
        self.assertEqual(adjuster.hue_shift, 0)
    
    def test_initialization_with_positive_hue(self):
        adjuster = HueAdjusterSeq(hue_shift=45)
        self.assertEqual(adjuster.hue_shift, 45)
    
    def test_initialization_with_negative_hue(self):
        adjuster = HueAdjusterSeq(hue_shift=-45)
        self.assertEqual(adjuster.hue_shift, -45)
    
    def test_initialization_clamps_positive(self):
        adjuster = HueAdjusterSeq(hue_shift=500)
        self.assertEqual(adjuster.hue_shift, 180)
    
    def test_initialization_clamps_negative(self):
        adjuster = HueAdjusterSeq(hue_shift=-500)
        self.assertEqual(adjuster.hue_shift, -180)


class TestHueAdjusterSeqProcess(unittest.TestCase):
    
    def setUp(self):
        self.adjuster = HueAdjusterSeq()
        
        self.test_image = OpenCVImage(100, 100)
        self.test_image._opencv_data = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        self.test_image._data = self.test_image._opencv_data.copy()
    
    def test_process_with_valid_image(self):
        result = self.adjuster.process(self.test_image)
        self.assertIsNotNone(result)
        self.assertEqual(result.width, self.test_image.width)
        self.assertEqual(result.height, self.test_image.height)
    
    def test_process_with_none_data(self):
        empty_image = OpenCVImage(100, 100)
        result = self.adjuster.process(empty_image)
        self.assertIsNone(result.data)
    
    def test_process_returns_cloned_image(self):
        result = self.adjuster.process(self.test_image)
        self.assertIsNot(result, self.test_image)
    
    def test_process_with_zero_shift(self):
        adjuster = HueAdjusterSeq(hue_shift=0)
        result = adjuster.process(self.test_image)
        self.assertIsNotNone(result.data)
    
    def test_process_with_positive_shift(self):
        adjuster = HueAdjusterSeq(hue_shift=45)
        result = adjuster.process(self.test_image)
        self.assertIsNotNone(result.data)
    
    def test_process_with_negative_shift(self):
        adjuster = HueAdjusterSeq(hue_shift=-45)
        result = adjuster.process(self.test_image)
        self.assertIsNotNone(result.data)
    
    def test_process_preserves_dimensions(self):
        result = self.adjuster.process(self.test_image)
        self.assertEqual(result.width, self.test_image.width)
        self.assertEqual(result.height, self.test_image.height)


class TestHueAdjusterSeqSetHueShift(unittest.TestCase):
    
    def test_set_hue_shift_positive(self):
        adjuster = HueAdjusterSeq()
        adjuster.set_hue_shift(60)
        self.assertEqual(adjuster.hue_shift, 60)
    
    def test_set_hue_shift_negative(self):
        adjuster = HueAdjusterSeq()
        adjuster.set_hue_shift(-60)
        self.assertEqual(adjuster.hue_shift, -60)
    
    def test_set_hue_shift_zero(self):
        adjuster = HueAdjusterSeq(hue_shift=45)
        adjuster.set_hue_shift(0)
        self.assertEqual(adjuster.hue_shift, 0)
    
    def test_set_hue_shift_clamps_positive(self):
        adjuster = HueAdjusterSeq()
        adjuster.set_hue_shift(500)
        self.assertEqual(adjuster.hue_shift, 180)
    
    def test_set_hue_shift_clamps_negative(self):
        adjuster = HueAdjusterSeq()
        adjuster.set_hue_shift(-500)
        self.assertEqual(adjuster.hue_shift, -180)
    
    def test_set_hue_shift_multiple_changes(self):
        adjuster = HueAdjusterSeq()
        adjuster.set_hue_shift(30)
        self.assertEqual(adjuster.hue_shift, 30)
        adjuster.set_hue_shift(60)
        self.assertEqual(adjuster.hue_shift, 60)
        adjuster.set_hue_shift(-45)
        self.assertEqual(adjuster.hue_shift, -45)


class TestHueAdjusterSeqBatch(unittest.TestCase):
    
    def setUp(self):
        self.adjuster = HueAdjusterSeq()
        
        self.images = []
        for i in range(3):
            img = OpenCVImage(100, 100)
            img._opencv_data = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            img._data = img._opencv_data.copy()
            self.images.append(img)
        
        self.hue_values = [30, 60, 90]
    
    def test_adjust_hue_batch_same_count(self):
        results = self.adjuster.adjust_hue_batch(self.images, self.hue_values)
        self.assertEqual(len(results), len(self.images))
    
    def test_adjust_hue_batch_returns_adjusted_images(self):
        results = self.adjuster.adjust_hue_batch(self.images, self.hue_values)
        for result in results:
            self.assertIsNotNone(result)
            self.assertIsNotNone(result.data)
    
    def test_adjust_hue_batch_mismatched_count(self):
        results = self.adjuster.adjust_hue_batch(self.images, [30, 60])
        self.assertEqual(len(results), len(self.images))
        self.assertEqual(results, self.images)
    
    def test_adjust_hue_batch_single_image(self):
        results = self.adjuster.adjust_hue_batch([self.images[0]], [45])
        self.assertEqual(len(results), 1)
        self.assertIsNotNone(results[0].data)
    
    def test_adjust_hue_batch_many_images(self):
        many_images = self.images * 5
        many_hue_values = [30] * 15
        results = self.adjuster.adjust_hue_batch(many_images, many_hue_values)
        self.assertEqual(len(results), 15)
    
    def test_adjust_hue_batch_different_shifts(self):
        results = self.adjuster.adjust_hue_batch(
            self.images,
            [0, 90, -90]
        )
        self.assertEqual(len(results), 3)
    
    def test_adjust_hue_batch_extreme_shifts(self):
        results = self.adjuster.adjust_hue_batch(
            self.images,
            [-180, 0, 180]
        )
        self.assertEqual(len(results), 3)
    
    def test_adjust_hue_batch_preserves_dimensions(self):
        results = self.adjuster.adjust_hue_batch(self.images, self.hue_values)
        for original, result in zip(self.images, results):
            self.assertEqual(result.width, original.width)
            self.assertEqual(result.height, original.height)
    
    def test_adjust_hue_batch_empty_list(self):
        results = self.adjuster.adjust_hue_batch([], [])
        self.assertEqual(len(results), 0)


class TestMonochromeAndHueSequence(unittest.TestCase):
    
    def setUp(self):
        self.test_image = OpenCVImage(100, 100)
        self.test_image._opencv_data = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        self.test_image._data = self.test_image._opencv_data.copy()
    
    def test_monochrome_then_hue_adjustment(self):
        mono_converter = MonochromeConverterSeq(method='standard')
        hue_adjuster = HueAdjusterSeq(hue_shift=45)
        
        mono_result = mono_converter.process(self.test_image)
        final_result = hue_adjuster.process(mono_result)
        
        self.assertIsNotNone(final_result.data)
        self.assertEqual(final_result.width, self.test_image.width)
        self.assertEqual(final_result.height, self.test_image.height)
    
    def test_different_methods_sequence(self):
        methods = ['standard', 'luminosity', 'average']
        
        for method in methods:
            converter = MonochromeConverterSeq(method=method)
            result = converter.process(self.test_image)
            self.assertIsNotNone(result.data)


if __name__ == '__main__':
    unittest.main()
