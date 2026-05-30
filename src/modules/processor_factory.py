##\file processor_factory.py
##\brief Factory pattern implementation for creating image processors
##\details Provides factory methods to create monochrome converters and hue adjusters
##\author Lab Team
##\version 1.0

from typing import Union, List
from enum import Enum

from ..image import OpenCVImage
from .sequential import MonochromeConverterSeq, HueAdjusterSeq
from .parallel import MonochromeConverterPar, HueAdjusterPar


##\brief Enumeration of processing modes
class ProcessingMode(Enum):
    SEQUENTIAL = "sequential" ##\brief Sequential processing mode
    PARALLEL = "parallel"     ##\brief Parallel processing mode


##\brief Factory class for creating image processors
##\details Implements factory pattern for creating converters and adjusters
class ProcessorFactory:
    
    @staticmethod
    def create_monochrome_converter(
        mode: ProcessingMode = ProcessingMode.SEQUENTIAL,
        method: str = "standard",
        num_threads: int = 4
    ) -> Union[MonochromeConverterSeq, MonochromeConverterPar]:
        ##\brief Create a monochrome converter
        ##\param mode Processing mode (SEQUENTIAL or PARALLEL)
        ##\param method Conversion method ("standard", "luminosity", or "average")
        ##\param num_threads Number of threads for parallel processing
        ##\return MonochromeConverterSeq or MonochromeConverterPar instance
        if mode == ProcessingMode.PARALLEL:
            return MonochromeConverterPar(method=method, num_threads=num_threads)
        else:
            return MonochromeConverterSeq(method=method)
    
    @staticmethod
    def create_hue_adjuster(
        mode: ProcessingMode = ProcessingMode.SEQUENTIAL,
        num_threads: int = 4
    ) -> Union[HueAdjusterSeq, HueAdjusterPar]:
        ##\brief Create a hue adjuster
        ##\param mode Processing mode (SEQUENTIAL or PARALLEL)
        ##\param num_threads Number of threads for parallel processing
        ##\return HueAdjusterSeq or HueAdjusterPar instance
        if mode == ProcessingMode.PARALLEL:
            return HueAdjusterPar(num_threads=num_threads)
        else:
            return HueAdjusterSeq()
