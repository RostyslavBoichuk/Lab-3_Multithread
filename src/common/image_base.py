##\file image_base.py
##\brief Abstract base class for image processing
##\details Defines the ImageBase abstract class and ImageFormat enum for image operations
##\author Lab Team
##\version 1.0

from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, Optional
import numpy as np


##\brief Enumeration of supported image color formats
class ImageFormat(Enum):
    BGR = "BGR"     ##\brief Blue-Green-Red format (OpenCV default)
    RGB = "RGB"     ##\brief Red-Green-Blue format
    GRAYSCALE = "GRAYSCALE" ##\brief Single channel grayscale
    HSV = "HSV"     ##\brief Hue-Saturation-Value format


##\brief Abstract base class for image operations
##\details Provides interface for image loading, saving, format conversion, and resizing
class ImageBase(ABC):
    
    def __init__(self, width: int, height: int, format: ImageFormat = ImageFormat.BGR):
        ##\brief Initialize base image object
        ##\param width Image width in pixels
        ##\param height Image height in pixels
        ##\param format Color format (default: BGR)
        self.width = width
        self.height = height
        self.format = format
        self._data: Optional[np.ndarray] = None
    
    @property
    def data(self) -> Optional[np.ndarray]:
        ##\brief Get image data as numpy array
        ##\return Numpy array with image data or None
        return self._data
    
    @abstractmethod
    def load(self, filepath: str) -> bool:
        ##\brief Load image from file
        ##\param filepath Path to image file
        ##\return True if successful, False otherwise
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> bool:
        ##\brief Save image to file
        ##\param filepath Path where to save image
        ##\return True if successful, False otherwise
        pass
    
    @abstractmethod
    def convert_format(self, target_format: ImageFormat) -> 'ImageBase':
        ##\brief Convert image to different color format
        ##\param target_format Target color format
        ##\return New ImageBase object with converted format
        pass
    
    @abstractmethod
    def resize(self, new_width: int, new_height: int) -> 'ImageBase':
        ##\brief Resize image to new dimensions
        ##\param new_width New width in pixels
        ##\param new_height New height in pixels
        ##\return New ImageBase object with resized image
        pass
    
    def get_dimensions(self) -> Tuple[int, int]:
        ##\brief Get image dimensions
        ##\return Tuple of (width, height)
        return (self.width, self.height)
    
    @abstractmethod
    def clone(self) -> 'ImageBase':
        ##\brief Create a copy of the image
        ##\return New ImageBase object with copied data
        pass
