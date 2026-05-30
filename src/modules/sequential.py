##\file sequential.py
##\brief Sequential image processing implementations
##\details Provides sequential (non-parallel) processors for monochrome conversion and hue adjustment
##\author Lab Team
##\version 1.0

import cv2
import numpy as np
import time
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod

from ..image import OpenCVImage
from .image_utils import (
    convert_to_monochrome, adjust_hue, clamp_hue_shift,
    SUPPORTED_MONO_METHODS
)


##\brief Abstract base class for image processors
class ProcessorBase(ABC):
    ##\brief Initialize processor
    def __init__(self):
        pass
    
    ##\brief Process single image
    ##\param image OpenCVImage to process
    ##\return Processed OpenCVImage
    @abstractmethod
    def process(self, image: OpenCVImage) -> OpenCVImage:
        pass


##\brief Sequential monochrome converter
##\details Converts images to monochrome sequentially (single-threaded)
class MonochromeConverterSeq(ProcessorBase):
    ##\brief Initialize converter
    ##\param method Conversion method ("standard", "luminosity", "average")
    def __init__(self, method: str = "standard"):
        super().__init__()
        self.method = method if method in SUPPORTED_MONO_METHODS else 'standard'
    
    def process(self, image: OpenCVImage) -> OpenCVImage:
        ##\brief Convert image to monochrome
        ##\param image OpenCVImage to convert
        ##\return New OpenCVImage in monochrome
        try:
            if image.data is None:
                return image
            
            result_data = convert_to_monochrome(image.data, self.method)
            
            result = image.clone()
            result._opencv_data = result_data
            result._data = result_data.copy()
            
            return result
            
        except Exception as e:
            return image


##\brief Sequential hue adjuster
##\details Adjusts image hue values sequentially (single-threaded)
class HueAdjusterSeq(ProcessorBase):
    ##\brief Initialize adjuster
    ##\param hue_shift Hue shift in degrees (-180 to 180)
    def __init__(self, hue_shift: int = 0):
        super().__init__()
        self.hue_shift = clamp_hue_shift(hue_shift)
    
    def set_hue_shift(self, hue_shift: int) -> None:
        ##\brief Set hue shift value
        ##\param hue_shift Hue shift in degrees (-180 to 180)
        self.hue_shift = clamp_hue_shift(hue_shift)
    
    def process(self, image: OpenCVImage) -> OpenCVImage:
        ##\brief Adjust image hue
        ##\param image OpenCVImage to process
        ##\return New OpenCVImage with adjusted hue
        try:
            if image.data is None:
                return image
            
            result_data = adjust_hue(image.data, self.hue_shift)
            
            result = image.clone()
            result._opencv_data = result_data
            result._data = result_data.copy()
            
            return result
            
        except Exception as e:
            return image
    
    def adjust_hue_batch(self, images: List[OpenCVImage], 
                         hue_values: List[int]) -> List[OpenCVImage]:
        ##\brief Process multiple images with varying hue values
        ##\param images List of OpenCVImage objects
        ##\param hue_values List of hue shift values
        ##\return List of processed OpenCVImage objects
        try:
            if len(images) != len(hue_values):
                return images
            
            adjusted_images = []
            for img, hue in zip(images, hue_values):
                self.set_hue_shift(hue)
                adjusted = self.process(img)
                adjusted_images.append(adjusted)
            
            return adjusted_images
            
        except Exception as e:
            return images
