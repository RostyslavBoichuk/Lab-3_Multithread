##\file test_parallel.py
##\brief Tests for parallel image processors
##\details Unit tests for parallel monochrome conversion and hue adjustment
##\author Lab Team
##\version 1.0

import unittest
import numpy as np
import cv2
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.modules.parallel import MonochromeConverterPar, HueAdjusterPar
from src.image.opencv_image import OpenCVImage
from src.modules.image_utils import SUPPORTED_MONO_METHODS


##\brief Test cases for MonochromeConverterPar initialization
class TestMonochromeConverterParInitialization(unittest.TestCase):
    
    def test_default_initialization(self):
        converter = MonochromeConverterPar()
        self.assertEqual(converter.method, 'standard')
        self.assertEqual(converter.num_threads, 4)
    
    def test_initialization_with_method(self):
        for method in SUPPORTED_MONO_METHODS:
            converter = MonochromeConverterPar(method=method)
            self.assertEqual(converter.method, method)
    
    def test_initialization_with_threads(self):
        for num_threads in [1, 2, 4, 8]:
            converter = MonochromeConverterPar(num_threads=num_threads)
            self.assertEqual(converter.num_threads, num_threads)
    
    def test_initialization_with_method_and_threads(self):
        converter = MonochromeConverterPar(method='luminosity', num_threads=2)
        self.assertEqual(converter.method, 'luminosity')
        self.assertEqual(converter.num_threads, 2)


##\brief Test cases for parallel monochrome batch processing
class TestMonochromeConverterParProcessBatch(unittest.TestCase):
    
    def setUp(self):
        self.converter = MonochromeConverterPar(num_threads=2)
        
        self.images = []
        for i in range(4):
            img = OpenCVImage(50, 50)
            img._opencv_data = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
            img._data = img._opencv_data.copy()
            self.images.append(img)
    
    def test_process_batch_single_image(self):
        results = self.converter.process_batch([self.images[0]])
        self.assertEqual(len(results), 1)
        self.assertIsNotNone(results[0])
    
    def test_process_batch_multiple_images(self):
        results = self.converter.process_batch(self.images)
        self.assertEqual(len(results), len(self.images))
    
    def test_process_batch_returns_images(self):
        results = self.converter.process_batch(self.images)
        for result in results:
            self.assertIsInstance(result, OpenCVImage)
    
    def test_process_batch_preserves_dimensions(self):
        results = self.converter.process_batch(self.images)
        for original, result in zip(self.images, results):
            self.assertEqual(result.width, original.width)
            self.assertEqual(result.height, original.height)
    
    def test_process_batch_standard_method(self):
        converter = MonochromeConverterPar(method='standard', num_threads=2)
        results = converter.process_batch(self.images)
        self.assertEqual(len(results), len(self.images))
    
    def test_process_batch_luminosity_method(self):
        converter = MonochromeConverterPar(method='luminosity', num_threads=2)
        results = converter.process_batch(self.images)
        self.assertEqual(len(results), len(self.images))
    
    def test_process_batch_average_method(self):
        converter = MonochromeConverterPar(method='average', num_threads=2)
        results = converter.process_batch(self.images)
        self.assertEqual(len(results), len(self.images))
    
    def test_process_batch_single_thread(self):
        converter = MonochromeConverterPar(num_threads=1)
        results = converter.process_batch(self.images)
        self.assertEqual(len(results), len(self.images))
    
    def test_process_batch_many_threads(self):
        converter = MonochromeConverterPar(num_threads=8)
        results = converter.process_batch(self.images)
        self.assertEqual(len(results), len(self.images))
    
    def test_process_batch_empty_list(self):
        results = self.converter.process_batch([])
        self.assertEqual(len(results), 0)
    
    def test_process_batch_many_images(self):
        many_images = self.images * 5
        results = self.converter.process_batch(many_images)
        self.assertEqual(len(results), 20)
    
    def test_process_batch_with_none_data_images(self):
        mixed_images = self.images.copy()
        empty_img = OpenCVImage(50, 50)
        mixed_images.append(empty_img)
        
        results = self.converter.process_batch(mixed_images)
        self.assertEqual(len(results), len(mixed_images))


class TestHueAdjusterParInitialization(unittest.TestCase):
    
    def test_default_initialization(self):
        adjuster = HueAdjusterPar()
        self.assertEqual(adjuster.num_threads, 4)
    
    def test_initialization_with_threads(self):
        for num_threads in [1, 2, 4, 8]:
            adjuster = HueAdjusterPar(num_threads=num_threads)
            self.assertEqual(adjuster.num_threads, num_threads)


class TestHueAdjusterParAdjustBatch(unittest.TestCase):
    
    def setUp(self):
        self.adjuster = HueAdjusterPar(num_threads=2)
        
        self.images = []
        for i in range(4):
            img = OpenCVImage(50, 50)
            img._opencv_data = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
            img._data = img._opencv_data.copy()
            self.images.append(img)
    
    def test_adjust_hue_batch_single_image(self):
        results = self.adjuster.adjust_hue_batch([self.images[0]], 45)
        self.assertEqual(len(results), 1)
        self.assertIsNotNone(results[0])
    
    def test_adjust_hue_batch_multiple_images(self):
        results = self.adjuster.adjust_hue_batch(self.images, 45)
        self.assertEqual(len(results), len(self.images))
    
    def test_adjust_hue_batch_zero_shift(self):
        results = self.adjuster.adjust_hue_batch(self.images, 0)
        self.assertEqual(len(results), len(self.images))
    
    def test_adjust_hue_batch_positive_shift(self):
        results = self.adjuster.adjust_hue_batch(self.images, 45)
        self.assertEqual(len(results), len(self.images))
    
    def test_adjust_hue_batch_negative_shift(self):
        results = self.adjuster.adjust_hue_batch(self.images, -45)
        self.assertEqual(len(results), len(self.images))
    
    def test_adjust_hue_batch_max_positive_shift(self):
        results = self.adjuster.adjust_hue_batch(self.images, 180)
        self.assertEqual(len(results), len(self.images))
    
    def test_adjust_hue_batch_max_negative_shift(self):
        results = self.adjuster.adjust_hue_batch(self.images, -180)
        self.assertEqual(len(results), len(self.images))
    
    def test_adjust_hue_batch_clamped_shift(self):
        results = self.adjuster.adjust_hue_batch(self.images, 500)
        self.assertEqual(len(results), len(self.images))
    
    def test_adjust_hue_batch_preserves_dimensions(self):
        results = self.adjuster.adjust_hue_batch(self.images, 60)
        for original, result in zip(self.images, results):
            self.assertEqual(result.width, original.width)
            self.assertEqual(result.height, original.height)
    
    def test_adjust_hue_batch_single_thread(self):
        adjuster = HueAdjusterPar(num_threads=1)
        results = adjuster.adjust_hue_batch(self.images, 45)
        self.assertEqual(len(results), len(self.images))
    
    def test_adjust_hue_batch_many_threads(self):
        adjuster = HueAdjusterPar(num_threads=8)
        results = adjuster.adjust_hue_batch(self.images, 45)
        self.assertEqual(len(results), len(self.images))
    
    def test_adjust_hue_batch_empty_list(self):
        results = self.adjuster.adjust_hue_batch([], 45)
        self.assertEqual(len(results), 0)
    
    def test_adjust_hue_batch_many_images(self):
        many_images = self.images * 5
        results = self.adjuster.adjust_hue_batch(many_images, 60)
        self.assertEqual(len(results), 20)


class TestHueAdjusterParVarying(unittest.TestCase):
    
    def setUp(self):
        self.adjuster = HueAdjusterPar(num_threads=2)
        
        self.images = []
        for i in range(3):
            img = OpenCVImage(50, 50)
            img._opencv_data = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
            img._data = img._opencv_data.copy()
            self.images.append(img)
    
    def test_adjust_hue_varying_same_count(self):
        hue_shifts = [30, 60, 90]
        results = self.adjuster.adjust_hue_varying(self.images, hue_shifts)
        self.assertEqual(len(results), len(self.images))
    
    def test_adjust_hue_varying_returns_images(self):
        hue_shifts = [30, 60, 90]
        results = self.adjuster.adjust_hue_varying(self.images, hue_shifts)
        for result in results:
            self.assertIsNotNone(result)
            self.assertIsNotNone(result.data)
    
    def test_adjust_hue_varying_mismatched_count(self):
        hue_shifts = [30, 60]
        results = self.adjuster.adjust_hue_varying(self.images, hue_shifts)
        self.assertEqual(results, self.images)
    
    def test_adjust_hue_varying_zero_shifts(self):
        hue_shifts = [0, 0, 0]
        results = self.adjuster.adjust_hue_varying(self.images, hue_shifts)
        self.assertEqual(len(results), 3)
    
    def test_adjust_hue_varying_negative_shifts(self):
        hue_shifts = [-30, -60, -90]
        results = self.adjuster.adjust_hue_varying(self.images, hue_shifts)
        self.assertEqual(len(results), 3)
    
    def test_adjust_hue_varying_mixed_shifts(self):
        hue_shifts = [-45, 0, 45]
        results = self.adjuster.adjust_hue_varying(self.images, hue_shifts)
        self.assertEqual(len(results), 3)
    
    def test_adjust_hue_varying_extreme_shifts(self):
        hue_shifts = [-180, 0, 180]
        results = self.adjuster.adjust_hue_varying(self.images, hue_shifts)
        self.assertEqual(len(results), 3)
    
    def test_adjust_hue_varying_single_image(self):
        results = self.adjuster.adjust_hue_varying([self.images[0]], [45])
        self.assertEqual(len(results), 1)
    
    def test_adjust_hue_varying_preserves_dimensions(self):
        hue_shifts = [30, 60, 90]
        results = self.adjuster.adjust_hue_varying(self.images, hue_shifts)
        for original, result in zip(self.images, results):
            self.assertEqual(result.width, original.width)
            self.assertEqual(result.height, original.height)
    
    def test_adjust_hue_varying_single_thread(self):
        adjuster = HueAdjusterPar(num_threads=1)
        hue_shifts = [30, 60, 90]
        results = adjuster.adjust_hue_varying(self.images, hue_shifts)
        self.assertEqual(len(results), 3)
    
    def test_adjust_hue_varying_many_threads(self):
        adjuster = HueAdjusterPar(num_threads=8)
        hue_shifts = [30, 60, 90]
        results = adjuster.adjust_hue_varying(self.images, hue_shifts)
        self.assertEqual(len(results), 3)


class TestParallelProcessingCorrectness(unittest.TestCase):
    
    def setUp(self):
        self.test_image = OpenCVImage(50, 50)
        self.test_image._opencv_data = np.ones((50, 50, 3), dtype=np.uint8) * 128
        self.test_image._data = self.test_image._opencv_data.copy()
    
    def test_monochrome_batch_consistency(self):
        images = [self.test_image.clone() for _ in range(3)]
        
        converter_par = MonochromeConverterPar(method='standard', num_threads=2)
        results = converter_par.process_batch(images)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsNotNone(result.data)
    
    def test_hue_batch_consistency(self):
        images = [self.test_image.clone() for _ in range(3)]
        
        adjuster = HueAdjusterPar(num_threads=2)
        results = adjuster.adjust_hue_batch(images, 45)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsNotNone(result.data)
    
    def test_hue_varying_consistency(self):
        images = [self.test_image.clone() for _ in range(3)]
        hue_shifts = [30, 60, 90]
        
        adjuster = HueAdjusterPar(num_threads=2)
        results = adjuster.adjust_hue_varying(images, hue_shifts)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsNotNone(result.data)


class TestParallelVsThread(unittest.TestCase):
    
    def setUp(self):
        self.images = []
        for i in range(5):
            img = OpenCVImage(50, 50)
            img._opencv_data = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
            img._data = img._opencv_data.copy()
            self.images.append(img)
    
    def test_single_vs_multiple_threads_monochrome(self):
        for threads in [1, 2, 4]:
            converter = MonochromeConverterPar(method='standard', num_threads=threads)
            results = converter.process_batch(self.images)
            self.assertEqual(len(results), len(self.images))
    
    def test_single_vs_multiple_threads_hue(self):
        for threads in [1, 2, 4]:
            adjuster = HueAdjusterPar(num_threads=threads)
            results = adjuster.adjust_hue_batch(self.images, 45)
            self.assertEqual(len(results), len(self.images))


if __name__ == '__main__':
    unittest.main()
