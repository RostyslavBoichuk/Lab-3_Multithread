##\file test_image_base.py
##\brief Tests for image base classes
##\details Unit tests for ImageFormat enum and ImageBase abstract class
##\author Lab Team
##\version 1.0

import unittest
from enum import Enum
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.image_base import ImageFormat, ImageBase


##\brief Test cases for ImageFormat enumeration
class TestImageFormat(unittest.TestCase):
    
    def test_image_format_values(self):
        self.assertEqual(ImageFormat.BGR.value, "BGR")
        self.assertEqual(ImageFormat.RGB.value, "RGB")
        self.assertEqual(ImageFormat.GRAYSCALE.value, "GRAYSCALE")
        self.assertEqual(ImageFormat.HSV.value, "HSV")
    
    def test_image_format_enum_members(self):
        formats = list(ImageFormat)
        self.assertEqual(len(formats), 4)
        self.assertIn(ImageFormat.BGR, formats)
        self.assertIn(ImageFormat.RGB, formats)
        self.assertIn(ImageFormat.GRAYSCALE, formats)
        self.assertIn(ImageFormat.HSV, formats)
    
    def test_image_format_names(self):
        self.assertEqual(ImageFormat.BGR.name, "BGR")
        self.assertEqual(ImageFormat.RGB.name, "RGB")
        self.assertEqual(ImageFormat.GRAYSCALE.name, "GRAYSCALE")
        self.assertEqual(ImageFormat.HSV.name, "HSV")


##\brief Concrete implementation of ImageBase for testing
class ConcreteImageForTesting(ImageBase):
    
    def load(self, filepath: str) -> bool:
        return True
    
    def save(self, filepath: str) -> bool:
        return True
    
    def convert_format(self, target_format: ImageFormat) -> 'ConcreteImageForTesting':
        new_image = ConcreteImageForTesting(self.width, self.height, target_format)
        new_image._data = self._data.copy() if self._data is not None else None
        return new_image
    
    def resize(self, new_width: int, new_height: int) -> 'ConcreteImageForTesting':
        new_image = ConcreteImageForTesting(new_width, new_height, self.format)
        new_image._data = self._data.copy() if self._data is not None else None
        return new_image
    
    def clone(self) -> 'ConcreteImageForTesting':
        new_image = ConcreteImageForTesting(self.width, self.height, self.format)
        new_image._data = self._data.copy() if self._data is not None else None
        return new_image


##\brief Test cases for ImageBase class
class TestImageBase(unittest.TestCase):
    
    def test_image_initialization_default(self):
        img = ConcreteImageForTesting(100, 200)
        self.assertEqual(img.width, 100)
        self.assertEqual(img.height, 200)
        self.assertEqual(img.format, ImageFormat.BGR)
        self.assertIsNone(img.data)
    
    def test_image_initialization_with_format(self):
        img = ConcreteImageForTesting(320, 240, ImageFormat.RGB)
        self.assertEqual(img.width, 320)
        self.assertEqual(img.height, 240)
        self.assertEqual(img.format, ImageFormat.RGB)
    
    def test_image_initialization_grayscale(self):
        img = ConcreteImageForTesting(640, 480, ImageFormat.GRAYSCALE)
        self.assertEqual(img.format, ImageFormat.GRAYSCALE)
    
    def test_image_initialization_hsv(self):
        img = ConcreteImageForTesting(100, 100, ImageFormat.HSV)
        self.assertEqual(img.format, ImageFormat.HSV)
    
    def test_get_dimensions(self):
        img = ConcreteImageForTesting(640, 480)
        dims = img.get_dimensions()
        self.assertEqual(dims, (640, 480))
        self.assertIsInstance(dims, tuple)
        self.assertEqual(len(dims), 2)
    
    def test_get_dimensions_square(self):
        img = ConcreteImageForTesting(512, 512)
        dims = img.get_dimensions()
        self.assertEqual(dims[0], dims[1])
    
    def test_get_dimensions_small_image(self):
        img = ConcreteImageForTesting(1, 1)
        dims = img.get_dimensions()
        self.assertEqual(dims, (1, 1))
    
    def test_get_dimensions_large_image(self):
        img = ConcreteImageForTesting(4096, 2160)
        dims = img.get_dimensions()
        self.assertEqual(dims, (4096, 2160))
    
    def test_data_property_none(self):
        img = ConcreteImageForTesting(100, 100)
        self.assertIsNone(img.data)
    
    def test_data_property_with_numpy_array(self):
        img = ConcreteImageForTesting(100, 100)
        test_data = np.zeros((100, 100, 3), dtype=np.uint8)
        img._data = test_data
        np.testing.assert_array_equal(img.data, test_data)
    
    def test_multiple_image_instances(self):
        img1 = ConcreteImageForTesting(100, 100, ImageFormat.BGR)
        img2 = ConcreteImageForTesting(200, 200, ImageFormat.RGB)
        img3 = ConcreteImageForTesting(150, 150, ImageFormat.GRAYSCALE)
        
        self.assertNotEqual(img1.width, img2.width)
        self.assertNotEqual(img2.format, img3.format)
        self.assertEqual(img1.format, ImageFormat.BGR)
        self.assertEqual(img2.format, ImageFormat.RGB)
        self.assertEqual(img3.format, ImageFormat.GRAYSCALE)
    
    def test_clone_preserves_dimensions(self):
        original = ConcreteImageForTesting(256, 512)
        cloned = original.clone()
        self.assertEqual(cloned.width, original.width)
        self.assertEqual(cloned.height, original.height)
    
    def test_clone_preserves_format(self):
        original = ConcreteImageForTesting(100, 100, ImageFormat.HSV)
        cloned = original.clone()
        self.assertEqual(cloned.format, original.format)
    
    def test_convert_format_creates_new_instance(self):
        original = ConcreteImageForTesting(100, 100, ImageFormat.BGR)
        converted = original.convert_format(ImageFormat.RGB)
        self.assertEqual(converted.format, ImageFormat.RGB)
        self.assertEqual(original.format, ImageFormat.BGR)
    
    def test_resize_creates_new_instance(self):
        original = ConcreteImageForTesting(100, 100)
        resized = original.resize(200, 200)
        self.assertEqual(resized.width, 200)
        self.assertEqual(resized.height, 200)
        self.assertEqual(original.width, 100)
        self.assertEqual(original.height, 100)
    
    def test_abstract_methods_not_callable(self):
        with self.assertRaises(TypeError):
            ImageBase(100, 100)


class TestImageBaseEdgeCases(unittest.TestCase):
    
    def test_zero_dimensions(self):
        img = ConcreteImageForTesting(0, 0)
        self.assertEqual(img.width, 0)
        self.assertEqual(img.height, 0)
    
    def test_very_large_dimensions(self):
        img = ConcreteImageForTesting(10000, 10000)
        self.assertEqual(img.width, 10000)
        self.assertEqual(img.height, 10000)
    
    def test_asymmetric_dimensions(self):
        img = ConcreteImageForTesting(1, 10000)
        self.assertEqual(img.width, 1)
        self.assertEqual(img.height, 10000)


if __name__ == '__main__':
    unittest.main()
