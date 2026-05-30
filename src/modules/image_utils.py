##\file image_utils.py
##\brief Image processing utility functions
##\details Common functions for monochrome conversion and hue adjustment
##\author Lab Team
##\version 1.0

import cv2
import numpy as np
from typing import Tuple

##\brief Minimum hue shift value in degrees
HUE_MIN = -180
##\brief Maximum hue shift value in degrees
HUE_MAX = 180
##\brief Hue scale factor for HSV conversion
HUE_SCALE = 2
##\brief List of supported monochrome conversion methods
SUPPORTED_MONO_METHODS = ['standard', 'luminosity', 'average']


def clamp_hue_shift(hue_shift: int) -> int:
    ##\brief Clamp hue shift value to valid range
    ##\param hue_shift Original hue shift value
    ##\return Clamped value between HUE_MIN and HUE_MAX
    return max(HUE_MIN, min(HUE_MAX, hue_shift))


def convert_to_monochrome(image_data: np.ndarray, method: str = "standard") -> np.ndarray:
    ##\brief Convert color image to monochrome
    ##\param image_data Input image as numpy array (BGR format)
    ##\param method Conversion method: "standard", "luminosity", or "average"
    ##\return Monochrome image as numpy array (BGR format with all channels equal)
    ##\throws ValueError If image_data is None or conversion fails
    if image_data is None:
        raise ValueError("Image data is None")
    
    if method not in SUPPORTED_MONO_METHODS:
        method = 'standard'
    
    try:
        if method == 'standard':
            gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        elif method == 'luminosity':
            b, g, r = cv2.split(image_data)
            gray = cv2.convertScaleAbs(0.114 * b + 0.587 * g + 0.299 * r)
        else:
            gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return result
        
    except Exception as e:
        raise ValueError(f"Monochrome conversion failed: {str(e)}")


def adjust_hue(image_data: np.ndarray, hue_shift: int) -> np.ndarray:
    ##\brief Adjust hue of color image
    ##\param image_data Input image as numpy array (BGR format)
    ##\param hue_shift Hue shift value in degrees (-180 to 180)
    ##\return Image with adjusted hue (BGR format)
    ##\throws ValueError If image_data is None or adjustment fails
    if image_data is None:
        raise ValueError("Image data is None")
    
    hue_shift = clamp_hue_shift(hue_shift)
    
    try:
        hsv = cv2.cvtColor(image_data, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        h = cv2.add(h, hue_shift / HUE_SCALE)
        h = np.uint8(np.clip(h, 0, 180))
        hsv_adjusted = cv2.merge([h, s, v])
        result = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)
        return result
        
    except Exception as e:
        raise ValueError(f"Hue adjustment failed: {str(e)}")
