##\file parallel.py
##\brief Parallel image processing implementations
##\details Provides multi-threaded processors for monochrome conversion and hue adjustment
##\author Lab Team
##\version 1.0

import cv2
import numpy as np
import threading
import time
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

from ..image import OpenCVImage
from .image_utils import convert_to_monochrome, adjust_hue, clamp_hue_shift


##\brief Parallel monochrome converter
##\details Converts multiple images to monochrome using thread pool
class MonochromeConverterPar:
    ##\brief Initialize parallel converter
    ##\param method Conversion method ("standard", "luminosity", "average")
    ##\param num_threads Number of worker threads
    def __init__(self, method: str = "standard", num_threads: int = 4):
        self.method = method
        self.num_threads = num_threads
    
    def _convert_to_monochrome(self, image: OpenCVImage, index: int) -> Tuple[int, OpenCVImage]:
        ##\brief Worker method for converting single image
        ##\param image OpenCVImage to convert
        ##\param index Image index in batch
        ##\return Tuple of (index, processed image)
        try:
            result_data = convert_to_monochrome(image.data, self.method)
            
            result = image.clone()
            result._opencv_data = result_data
            result._data = result_data.copy()
            
            return index, result
        except Exception as e:
            return index, image
    
    def process_batch(self, images: List[OpenCVImage]) -> List[OpenCVImage]:
        ##\brief Convert batch of images in parallel
        ##\param images List of OpenCVImage objects
        ##\return List of converted images maintaining order
        try:
            results = [None] * len(images)
            
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = {
                    executor.submit(self._convert_to_monochrome, img, idx): idx
                    for idx, img in enumerate(images)
                }
                
                for future in as_completed(futures):
                    idx, converted = future.result()
                    results[idx] = converted
            
            return results
            
        except Exception as e:
            return []


##\brief Parallel hue adjuster
##\details Adjusts hue of multiple images using thread pool
class HueAdjusterPar:
    ##\brief Initialize parallel adjuster
    ##\param num_threads Number of worker threads
    def __init__(self, num_threads: int = 4):
        self.num_threads = num_threads
    
    def _adjust_hue(self, image: OpenCVImage, hue_shift: int, 
                    index: int) -> Tuple[int, OpenCVImage]:
        ##\brief Worker method for adjusting single image hue
        ##\param image OpenCVImage to process
        ##\param hue_shift Hue shift value
        ##\param index Image index in batch
        ##\return Tuple of (index, processed image)
        try:
            result_data = adjust_hue(image.data, hue_shift)
            
            result = image.clone()
            result._opencv_data = result_data
            result._data = result_data.copy()
            
            return index, result
        except Exception as e:
            return index, image
    
    def adjust_hue_batch(self, images: List[OpenCVImage], 
                        hue_shift: int) -> List[OpenCVImage]:
        ##\brief Adjust hue of image batch with same value in parallel
        ##\param images List of OpenCVImage objects
        ##\param hue_shift Common hue shift value for all images
        ##\return List of adjusted images maintaining order
        try:
            results = [None] * len(images)
            
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = {
                    executor.submit(self._adjust_hue, img, hue_shift, idx): idx
                    for idx, img in enumerate(images)
                }
                
                for future in as_completed(futures):
                    idx, adjusted = future.result()
                    results[idx] = adjusted
            
            return results
            
        except Exception as e:
            return []
    
    def adjust_hue_varying(self, images: List[OpenCVImage],
                          hue_shifts: List[int]) -> List[OpenCVImage]:
        ##\brief Adjust hue of image batch with varying values in parallel
        ##\param images List of OpenCVImage objects
        ##\param hue_shifts List of hue shift values for each image
        ##\return List of adjusted images maintaining order
        try:
            if len(images) != len(hue_shifts):
                return images
            
            results = [None] * len(images)
            
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = {
                    executor.submit(self._adjust_hue, img, hue, idx): idx
                    for idx, (img, hue) in enumerate(zip(images, hue_shifts))
                }
                
                for future in as_completed(futures):
                    idx, adjusted = future.result()
                    results[idx] = adjusted
            
            return results
            
        except Exception as e:
            return []

