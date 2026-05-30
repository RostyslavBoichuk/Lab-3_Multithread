##\file test_image_utils.py
##\brief Tests for image utility functions
##\details Unit tests for image processing utility functions
##\author Lab Team
##\version 1.0

import unittest
import numpy as np
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.modules.image_utils import (
    clamp_hue_shift, convert_to_monochrome, adjust_hue,
    HUE_MIN, HUE_MAX, HUE_SCALE, SUPPORTED_MONO_METHODS
)


##\brief Test cases for hue shift clamping
class TestHueClamp(unittest.TestCase):
    
    def test_clamp_within_range(self):
        result = clamp_hue_shift(0)
        self.assertEqual(result, 0)
        
        result = clamp_hue_shift(45)
        self.assertEqual(result, 45)
        
        result = clamp_hue_shift(-45)
        self.assertEqual(result, -45)
    
    def test_clamp_minimum_boundary(self):
        result = clamp_hue_shift(-180)
        self.assertEqual(result, -180)
    
    def test_clamp_maximum_boundary(self):
        result = clamp_hue_shift(180)
        self.assertEqual(result, 180)
    
    def test_clamp_below_minimum(self):
        result = clamp_hue_shift(-200)
        self.assertEqual(result, -180)
        
        result = clamp_hue_shift(-500)
        self.assertEqual(result, -180)
    
    def test_clamp_above_maximum(self):
        result = clamp_hue_shift(200)
        self.assertEqual(result, 180)
        
        result = clamp_hue_shift(500)
        self.assertEqual(result, 180)
    
    def test_clamp_zero(self):
        result = clamp_hue_shift(0)
        self.assertEqual(result, 0)
    
    def test_clamp_large_negative(self):
        result = clamp_hue_shift(-9999)
        self.assertEqual(result, HUE_MIN)
    
    def test_clamp_large_positive(self):
        result = clamp_hue_shift(9999)
        self.assertEqual(result, HUE_MAX)


##\brief Test cases for supported monochrome methods
class TestSupportedMethods(unittest.TestCase):
    
    def test_supported_methods_list(self):
        self.assertIsInstance(SUPPORTED_MONO_METHODS, list)
        self.assertGreater(len(SUPPORTED_MONO_METHODS), 0)
    
    def test_supported_methods_content(self):
        self.assertIn('standard', SUPPORTED_MONO_METHODS)
        self.assertIn('luminosity', SUPPORTED_MONO_METHODS)
        self.assertIn('average', SUPPORTED_MONO_METHODS)
    
    def test_hue_constants(self):
        self.assertEqual(HUE_MIN, -180)
        self.assertEqual(HUE_MAX, 180)
        self.assertEqual(HUE_SCALE, 2)


##\brief Test cases for monochrome conversion
class TestConvertToMonochrome(unittest.TestCase):
    
    def setUp(self):
        self.test_image = np.array([
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            [[255, 255, 0], [255, 0, 255], [0, 255, 255]],
            [[128, 128, 128], [64, 64, 64], [192, 192, 192]]
        ], dtype=np.uint8)
        
        self.large_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    def test_convert_standard_method(self):
        result = convert_to_monochrome(self.test_image, 'standard')
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (3, 3, 3))
    
    def test_convert_luminosity_method(self):
        result = convert_to_monochrome(self.test_image, 'luminosity')
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (3, 3, 3))
    
    def test_convert_average_method(self):
        result = convert_to_monochrome(self.test_image, 'average')
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (3, 3, 3))
    
    def test_convert_invalid_method_defaults_to_standard(self):
        result_invalid = convert_to_monochrome(self.test_image, 'invalid_method')
        result_standard = convert_to_monochrome(self.test_image, 'standard')
        np.testing.assert_array_equal(result_invalid, result_standard)
    
    def test_convert_large_image(self):
        result = convert_to_monochrome(self.large_image, 'standard')
        self.assertEqual(result.shape, (100, 100, 3))
    
    def test_convert_returns_numpy_array(self):
        result = convert_to_monochrome(self.test_image, 'standard')
        self.assertIsInstance(result, np.ndarray)
    
    def test_convert_maintains_dtype(self):
        result = convert_to_monochrome(self.test_image, 'standard')
        self.assertEqual(result.dtype, np.uint8)
    
    def test_convert_output_is_not_none(self):
        result = convert_to_monochrome(self.test_image, 'luminosity')
        self.assertIsNotNone(result)
    
    def test_convert_none_data_raises_error(self):
        with self.assertRaises(ValueError):
            convert_to_monochrome(None, 'standard')
    
    def test_convert_all_methods_produce_output(self):
        for method in SUPPORTED_MONO_METHODS:
            result = convert_to_monochrome(self.test_image, method)
            self.assertIsNotNone(result)
            self.assertEqual(result.shape, (3, 3, 3))
    
    def test_convert_single_pixel_image(self):
        small_image = np.array([[[100, 150, 200]]], dtype=np.uint8)
        result = convert_to_monochrome(small_image, 'standard')
        self.assertEqual(result.shape, (1, 1, 3))
    
    def test_convert_white_image(self):
        white_image = np.full((10, 10, 3), 255, dtype=np.uint8)
        result = convert_to_monochrome(white_image, 'standard')
        self.assertIsNotNone(result)
        self.assertTrue(np.all(result > 0))
    
    def test_convert_black_image(self):
        black_image = np.zeros((10, 10, 3), dtype=np.uint8)
        result = convert_to_monochrome(black_image, 'standard')
        self.assertIsNotNone(result)
        self.assertTrue(np.all(result == 0))


class TestAdjustHue(unittest.TestCase):
    
    def setUp(self):
        self.test_image = np.array([
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            [[255, 255, 0], [255, 0, 255], [0, 255, 255]],
            [[128, 128, 128], [64, 64, 64], [192, 192, 192]]
        ], dtype=np.uint8)
        
        self.large_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    def test_adjust_hue_zero_shift(self):
        result = adjust_hue(self.test_image, 0)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, self.test_image.shape)
    
    def test_adjust_hue_positive_shift(self):
        result = adjust_hue(self.test_image, 45)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, self.test_image.shape)
    
    def test_adjust_hue_negative_shift(self):
        result = adjust_hue(self.test_image, -45)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, self.test_image.shape)
    
    def test_adjust_hue_max_positive(self):
        result = adjust_hue(self.test_image, 180)
        self.assertIsNotNone(result)
    
    def test_adjust_hue_max_negative(self):
        result = adjust_hue(self.test_image, -180)
        self.assertIsNotNone(result)
    
    def test_adjust_hue_returns_numpy_array(self):
        result = adjust_hue(self.test_image, 30)
        self.assertIsInstance(result, np.ndarray)
    
    def test_adjust_hue_maintains_dtype(self):
        result = adjust_hue(self.test_image, 30)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_adjust_hue_maintains_shape(self):
        result = adjust_hue(self.large_image, 60)
        self.assertEqual(result.shape, self.large_image.shape)
    
    def test_adjust_hue_none_data_raises_error(self):
        with self.assertRaises(ValueError):
            adjust_hue(None, 45)
    
    def test_adjust_hue_over_clamped_value(self):
        result = adjust_hue(self.test_image, 500)
        self.assertIsNotNone(result)
    
    def test_adjust_hue_under_clamped_value(self):
        result = adjust_hue(self.test_image, -500)
        self.assertIsNotNone(result)
    
    def test_adjust_hue_single_pixel(self):
        small_image = np.array([[[100, 150, 200]]], dtype=np.uint8)
        result = adjust_hue(small_image, 45)
        self.assertEqual(result.shape, (1, 1, 3))
    
    def test_adjust_hue_white_image(self):
        white_image = np.full((10, 10, 3), 255, dtype=np.uint8)
        result = adjust_hue(white_image, 90)
        self.assertIsNotNone(result)
    
    def test_adjust_hue_gray_image(self):
        gray_image = np.full((10, 10, 3), 128, dtype=np.uint8)
        result = adjust_hue(gray_image, 90)
        self.assertIsNotNone(result)
    
    def test_adjust_hue_multiple_shifts(self):
        result_30 = adjust_hue(self.test_image, 30)
        result_60 = adjust_hue(self.test_image, 60)
        result_90 = adjust_hue(self.test_image, 90)
        
        self.assertFalse(np.array_equal(result_30, result_60))
        self.assertFalse(np.array_equal(result_60, result_90))


class TestImageUtilsEdgeCases(unittest.TestCase):
    
    def test_monochrome_grayscale_input(self):
        gray_like = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = convert_to_monochrome(gray_like, 'standard')
        self.assertEqual(result.shape, (100, 100, 3))
    
    def test_hue_adjustment_saturation_zero(self):
        desaturated = np.full((50, 50, 3), 100, dtype=np.uint8)
        result = adjust_hue(desaturated, 90)
        self.assertIsNotNone(result)
    
    def test_extreme_color_values(self):
        extreme = np.array([
            [[0, 0, 0], [255, 255, 255]],
            [[0, 255, 0], [255, 0, 255]]
        ], dtype=np.uint8)
        
        result_mono = convert_to_monochrome(extreme, 'luminosity')
        result_hue = adjust_hue(extreme, 45)
        
        self.assertIsNotNone(result_mono)
        self.assertIsNotNone(result_hue)


if __name__ == '__main__':
    unittest.main()
