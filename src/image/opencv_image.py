##\file opencv_image.py
##\brief OpenCV-based implementation of ImageBase
##\details Provides concrete image operations using OpenCV library
##\author Lab Team
##\version 1.0

import cv2
import numpy as np
from typing import Optional
from pathlib import Path

from ..common import ImageBase, ImageFormat


##\brief OpenCV implementation of ImageBase
##\details Uses OpenCV (cv2) library for image operations
class OpenCVImage(ImageBase):
    
    def __init__(self, width: int = 0, height: int = 0, 
                 format: ImageFormat = ImageFormat.BGR):
        ##\brief Initialize OpenCVImage object
        ##\param width Image width (default: 0)
        ##\param height Image height (default: 0)
        ##\param format Color format (default: BGR)
        super().__init__(width, height, format)
        self._opencv_data: Optional[np.ndarray] = None
    
    def load(self, filepath: str) -> bool:
        ##\brief Load image from file using OpenCV
        ##\param filepath Path to image file
        ##\return True if successful, False otherwise
        try:
            path = Path(filepath)
            if not path.exists():
                return False
            
            self._opencv_data = cv2.imread(str(path))
            
            if self._opencv_data is None:
                return False
            
            self.height, self.width = self._opencv_data.shape[:2]
            self._data = self._opencv_data.copy()
            
            return True
            
        except Exception as e:
            return False
    
    def save(self, filepath: str) -> bool:
        ##\brief Save image to file using OpenCV
        ##\param filepath Destination file path
        ##\return True if successful, False otherwise
        try:
            if self._opencv_data is None:
                return False
            
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            success = cv2.imwrite(str(path), self._opencv_data)
            
            return success
            
        except Exception as e:
            return False
    
    def convert_format(self, target_format: ImageFormat) -> 'OpenCVImage':
        ##\brief Convert image to different color format
        ##\param target_format Target ImageFormat enum value
        ##\return New OpenCVImage with converted format
        try:
            if self._opencv_data is None:
                return OpenCVImage()
            
            new_image = OpenCVImage(self.width, self.height, target_format)
            
            if target_format == ImageFormat.BGR:
                new_image._opencv_data = self._opencv_data.copy()
            elif target_format == ImageFormat.RGB:
                new_image._opencv_data = cv2.cvtColor(self._opencv_data, cv2.COLOR_BGR2RGB)
            elif target_format == ImageFormat.GRAYSCALE:
                new_image._opencv_data = cv2.cvtColor(self._opencv_data, cv2.COLOR_BGR2GRAY)
            elif target_format == ImageFormat.HSV:
                new_image._opencv_data = cv2.cvtColor(self._opencv_data, cv2.COLOR_BGR2HSV)
            
            new_image._data = new_image._opencv_data.copy()
            return new_image
            
        except Exception as e:
            return OpenCVImage()
    
    def resize(self, new_width: int, new_height: int) -> 'OpenCVImage':
        ##\brief Resize image to new dimensions
        ##\param new_width New width in pixels
        ##\param new_height New height in pixels
        ##\return New OpenCVImage with resized image
        try:
            if self._opencv_data is None:
                return OpenCVImage()
            
            resized = cv2.resize(self._opencv_data, (new_width, new_height))
            
            new_image = OpenCVImage(new_width, new_height, self.format)
            new_image._opencv_data = resized
            new_image._data = resized.copy()
            
            return new_image
            
        except Exception as e:
            return OpenCVImage()
    
    def clone(self) -> 'OpenCVImage':
        ##\brief Create a deep copy of the image
        ##\return New OpenCVImage with copied data
        try:
            if self._opencv_data is None:
                return OpenCVImage()
            
            new_image = OpenCVImage(self.width, self.height, self.format)
            new_image._opencv_data = self._opencv_data.copy()
            new_image._data = new_image._opencv_data.copy()
            
            return new_image
            
        except Exception as e:
            return OpenCVImage()
    
    def apply_gaussian_blur(self, kernel_size: int = 5) -> 'OpenCVImage':
        try:
            if self._opencv_data is None:
                return OpenCVImage()
            
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            blurred = cv2.GaussianBlur(self._opencv_data, (kernel_size, kernel_size), 0)
            
            new_image = OpenCVImage(self.width, self.height, self.format)
            new_image._opencv_data = blurred
            new_image._data = blurred.copy()
            
            return new_image
            
        except Exception as e:
            return OpenCVImage()