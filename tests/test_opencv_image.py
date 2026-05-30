##\file test_opencv_image.py
##\brief Tests for OpenCV image implementation
##\details Unit tests for OpenCVImage class functionality
##\author Lab Team
##\version 1.0

import unittest
import tempfile
import numpy as np
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.image.opencv_image import OpenCVImage
from src.common.image_base import ImageFormat


##\brief Test cases for OpenCVImage initialization
class TestOpenCVImageInitialization(unittest.TestCase):
    
    def test_default_initialization(self):
        img = OpenCVImage()
        self.assertEqual(img.width, 0)
        self.assertEqual(img.height, 0)
        self.assertEqual(img.format, ImageFormat.BGR)
        self.assertIsNone(img.data)
    
    def test_initialization_with_dimensions(self):
        img = OpenCVImage(640, 480)
        self.assertEqual(img.width, 640)
        self.assertEqual(img.height, 480)
    
    def test_initialization_with_format(self):
        img = OpenCVImage(320, 240, ImageFormat.RGB)
        self.assertEqual(img.format, ImageFormat.RGB)
    
    def test_initialization_grayscale(self):
        img = OpenCVImage(100, 100, ImageFormat.GRAYSCALE)
        self.assertEqual(img.format, ImageFormat.GRAYSCALE)


##\brief Test cases for OpenCVImage load functionality
class TestOpenCVImageLoad(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        test_image_bgr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        self.test_image_path = str(self.temp_path / "test_image.jpg")
        cv2.imwrite(self.test_image_path, test_image_bgr)
        
        self.test_image_png_path = str(self.temp_path / "test_image.png")
        cv2.imwrite(self.test_image_png_path, test_image_bgr)
    
    def tearDown(self):
        self.temp_dir.cleanup()
    
    def test_load_existing_image(self):
        img = OpenCVImage()
        success = img.load(self.test_image_path)
        self.assertTrue(success)
        self.assertEqual(img.width, 100)
        self.assertEqual(img.height, 100)
        self.assertIsNotNone(img.data)
    
    def test_load_nonexistent_image(self):
        img = OpenCVImage()
        success = img.load("/nonexistent/path/image.jpg")
        self.assertFalse(success)
    
    def test_load_invalid_image_path(self):
        img = OpenCVImage()
        success = img.load("")
        self.assertFalse(success)
    
    def test_load_updates_dimensions(self):
        img = OpenCVImage(0, 0)
        img.load(self.test_image_path)
        self.assertEqual(img.width, 100)
        self.assertEqual(img.height, 100)
    
    def test_load_png_format(self):
        img = OpenCVImage()
        success = img.load(self.test_image_png_path)
        self.assertTrue(success)
    
    def test_load_creates_data_copy(self):
        img = OpenCVImage()
        img.load(self.test_image_path)
        self.assertIsNotNone(img.data)
        self.assertIsNotNone(img._opencv_data)


class TestOpenCVImageSave(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
    
    def tearDown(self):
        self.temp_dir.cleanup()
    
    def test_save_image(self):
        img = OpenCVImage(100, 100)
        img._opencv_data = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        save_path = str(self.temp_path / "output.jpg")
        success = img.save(save_path)
        
        self.assertTrue(success)
        self.assertTrue(Path(save_path).exists())
    
    def test_save_without_data(self):
        img = OpenCVImage(100, 100)
        save_path = str(self.temp_path / "empty.jpg")
        success = img.save(save_path)
        self.assertFalse(success)
    
    def test_save_creates_directories(self):
        img = OpenCVImage(50, 50)
        img._opencv_data = np.zeros((50, 50, 3), dtype=np.uint8)
        
        nested_path = str(self.temp_path / "subdir" / "nested" / "output.jpg")
        success = img.save(nested_path)
        
        self.assertTrue(success)
        self.assertTrue(Path(nested_path).exists())
    
    def test_save_and_load_roundtrip(self):
        original_data = np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8)
        img = OpenCVImage(80, 80)
        img._opencv_data = original_data.copy()
        
        save_path = str(self.temp_path / "roundtrip.jpg")
        img.save(save_path)
        
        loaded_img = OpenCVImage()
        loaded_img.load(save_path)
        
        self.assertEqual(loaded_img.width, 80)
        self.assertEqual(loaded_img.height, 80)


class TestOpenCVImageConvertFormat(unittest.TestCase):
    
    def setUp(self):
        self.img = OpenCVImage(100, 100)
        self.img._opencv_data = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        self.img._data = self.img._opencv_data.copy()
    
    def test_convert_bgr_to_rgb(self):
        converted = self.img.convert_format(ImageFormat.RGB)
        self.assertEqual(converted.format, ImageFormat.RGB)
        self.assertEqual(converted.width, 100)
        self.assertEqual(converted.height, 100)
    
    def test_convert_bgr_to_grayscale(self):
        converted = self.img.convert_format(ImageFormat.GRAYSCALE)
        self.assertEqual(converted.format, ImageFormat.GRAYSCALE)
        self.assertIsNotNone(converted.data)
    
    def test_convert_bgr_to_hsv(self):
        converted = self.img.convert_format(ImageFormat.HSV)
        self.assertEqual(converted.format, ImageFormat.HSV)
        self.assertIsNotNone(converted.data)
    
    def test_convert_bgr_to_bgr(self):
        converted = self.img.convert_format(ImageFormat.BGR)
        self.assertEqual(converted.format, ImageFormat.BGR)
        self.assertIsNotNone(converted.data)
    
    def test_convert_without_data(self):
        empty_img = OpenCVImage(100, 100)
        converted = empty_img.convert_format(ImageFormat.RGB)
        self.assertIsNone(converted.data)
    
    def test_conversion_preserves_dimensions(self):
        converted = self.img.convert_format(ImageFormat.GRAYSCALE)
        self.assertEqual(converted.width, self.img.width)
        self.assertEqual(converted.height, self.img.height)
    
    def test_conversion_does_not_modify_original(self):
        original_format = self.img.format
        self.img.convert_format(ImageFormat.RGB)
        self.assertEqual(self.img.format, original_format)


class TestOpenCVImageResize(unittest.TestCase):
    
    def setUp(self):
        self.img = OpenCVImage(100, 100)
        self.img._opencv_data = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        self.img._data = self.img._opencv_data.copy()
    
    def test_resize_to_larger(self):
        resized = self.img.resize(200, 200)
        self.assertEqual(resized.width, 200)
        self.assertEqual(resized.height, 200)
    
    def test_resize_to_smaller(self):
        resized = self.img.resize(50, 50)
        self.assertEqual(resized.width, 50)
        self.assertEqual(resized.height, 50)
    
    def test_resize_asymmetric(self):
        resized = self.img.resize(200, 50)
        self.assertEqual(resized.width, 200)
        self.assertEqual(resized.height, 50)
    
    def test_resize_preserves_format(self):
        resized = self.img.resize(150, 150)
        self.assertEqual(resized.format, self.img.format)
    
    def test_resize_does_not_modify_original(self):
        original_width = self.img.width
        self.img.resize(200, 200)
        self.assertEqual(self.img.width, original_width)
    
    def test_resize_without_data(self):
        empty_img = OpenCVImage(100, 100)
        resized = empty_img.resize(200, 200)
        self.assertIsNone(resized.data)
    
    def test_resize_to_minimal(self):
        resized = self.img.resize(1, 1)
        self.assertEqual(resized.width, 1)
        self.assertEqual(resized.height, 1)


class TestOpenCVImageClone(unittest.TestCase):
    
    def setUp(self):
        self.img = OpenCVImage(100, 100, ImageFormat.RGB)
        self.img._opencv_data = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        self.img._data = self.img._opencv_data.copy()
    
    def test_clone_creates_new_instance(self):
        cloned = self.img.clone()
        self.assertIsNot(cloned, self.img)
    
    def test_clone_preserves_dimensions(self):
        cloned = self.img.clone()
        self.assertEqual(cloned.width, self.img.width)
        self.assertEqual(cloned.height, self.img.height)
    
    def test_clone_preserves_format(self):
        cloned = self.img.clone()
        self.assertEqual(cloned.format, self.img.format)
    
    def test_clone_copies_data(self):
        cloned = self.img.clone()
        self.assertIsNotNone(cloned.data)
        self.assertIsNot(cloned.data, self.img.data)
        np.testing.assert_array_equal(cloned.data, self.img.data)
    
    def test_clone_without_data(self):
        empty_img = OpenCVImage(100, 100)
        cloned = empty_img.clone()
        self.assertIsNone(cloned.data)
    
    def test_clone_independence(self):
        cloned = self.img.clone()
        cloned.width = 50
        self.assertEqual(self.img.width, 100)


class TestOpenCVImageGaussianBlur(unittest.TestCase):
    
    def setUp(self):
        self.img = OpenCVImage(100, 100)
        self.img._opencv_data = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        self.img._data = self.img._opencv_data.copy()
    
    def test_apply_gaussian_blur_default(self):
        blurred = self.img.apply_gaussian_blur()
        self.assertIsNotNone(blurred.data)
        self.assertEqual(blurred.width, self.img.width)
        self.assertEqual(blurred.height, self.img.height)
    
    def test_apply_gaussian_blur_custom_kernel(self):
        blurred = self.img.apply_gaussian_blur(kernel_size=7)
        self.assertIsNotNone(blurred.data)
    
    def test_apply_gaussian_blur_even_kernel(self):
        blurred = self.img.apply_gaussian_blur(kernel_size=4)
        self.assertIsNotNone(blurred.data)
    
    def test_apply_gaussian_blur_small_kernel(self):
        blurred = self.img.apply_gaussian_blur(kernel_size=3)
        self.assertIsNotNone(blurred.data)
    
    def test_apply_gaussian_blur_large_kernel(self):
        blurred = self.img.apply_gaussian_blur(kernel_size=51)
        self.assertIsNotNone(blurred.data)
    
    def test_apply_gaussian_blur_does_not_modify_original(self):
        original_data = self.img.data.copy()
        self.img.apply_gaussian_blur()
        np.testing.assert_array_equal(self.img.data, original_data)
    
    def test_apply_gaussian_blur_without_data(self):
        empty_img = OpenCVImage(100, 100)
        blurred = empty_img.apply_gaussian_blur()
        self.assertIsNone(blurred.data)


if __name__ == '__main__':
    unittest.main()
