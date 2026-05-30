# Image Color Processing Application - Lab 3

A sophisticated multithreaded Python application for real-time image color processing with both sequential and parallel implementations using modern concurrency patterns.

## Overview

This project demonstrates professional software engineering practices for parallel computing in Python, implementing two core image processing features:

- **Monochrome Conversion**: Convert color images to grayscale using three distinct methods (standard, luminosity, average)
- **Hue Adjustment**: Apply HSV-based hue shifting to images (range: -180° to +180°)

Both features are implemented with **sequential** and **parallel** variants, allowing direct performance comparison and showcasing multiprocessing benefits.

## Features

### Core Processing
- **Three Monochrome Methods**
  - Standard (BT.709 luminosity)
  - Luminosity-weighted
  - Average RGB channels
  
- **Hue Shifting** (HSV color space)
  - Batch processing (uniform hue shift)
  - Varying shifts (different shift per image)
  - Range clamping with safety validation

- **Parallel Processing**
  - ProcessPoolExecutor-based multiprocessing (true parallelism, bypasses Python GIL)
  - Configurable thread pool (1-8 workers)
  - Performance benchmarking with speedup metrics
  - Observed speedups: up to 6.77x on multi-core systems

### User Interfaces
- **GUI Application** (PyQt5)
  - Dual-tab interface: Monochrome Conversion & Hue Adjustment
  - Real-time preview
  - Multi-threaded background processing
  - Thread-safe signal/slot architecture

- **Console Application**
  - 10 demonstration examples
  - Feature showcase with timing metrics
  - Batch processing demo

### Quality Assurance
- **Comprehensive Test Suite** (217 unit tests)
  - Image I/O and format conversion
  - Color processing correctness
  - Parallel vs sequential validation
  - GUI thread safety (16 dedicated tests)
  - Edge case handling
  
- **Performance Benchmarking**
  - Benchmark suite with CSV results
  - Multi-method and multi-thread comparison
  - Speedup calculation and analysis

- **Professional Documentation**
  - Doxygen API documentation
  - Inline code documentation (Doxygen format)
  - 17+ documented files
  - Architecture diagrams

## Requirements

- Python 3.8+
- numpy >= 1.19.0
- opencv-python >= 4.5.0
- PyQt5 >= 5.15.0 (GUI only)

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Quick Start

### GUI Application
```bash
python gui_launcher.py
```
Then select **Option 2** for the GUI interface.

### Console Application  
```bash
python console_launcher.py
```
Select **Option 1** to run the console demo with 10 examples.

### Main Menu
```bash
python main.py
```
Navigate between console and GUI modes.

## Project Structure

```
Lab-3/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── main.py                           # Main application menu
├── console_launcher.py               # Console entry point
├── gui_launcher.py                   # GUI entry point
├── Doxyfile                          # Documentation configuration
│
├── src/                              # Source code
│   ├── common/
│   │   └── image_base.py            # Image format enum & abstract base
│   ├── image/
│   │   └── opencv_image.py          # OpenCV implementation
│   ├── modules/
│   │   ├── processor_factory.py      # Factory pattern for processors
│   │   ├── sequential.py            # Sequential implementations
│   │   ├── parallel.py              # Parallel implementations
│   │   └── image_utils.py           # Utility functions
│   └── gui.py                        # PyQt5 GUI implementation
│
├── tests/                            # Test suite (217 tests)
│   ├── run_all_tests.py             # Test runner
│   ├── test_image_base.py           # Image format tests (23 tests)
│   ├── test_opencv_image.py         # OpenCV tests (47 tests)
│   ├── test_image_utils.py          # Utility tests (38 tests)
│   ├── test_sequential.py           # Sequential processor tests (37 tests)
│   ├── test_parallel.py             # Parallel processor tests (56 tests)
│   └── test_gui_thread_safety.py    # GUI threading tests (16 tests)
│
├── benchmarks/                       # Performance benchmarking
│   ├── benchmark.py                 # Benchmark suite
│   ├── benchmark_results.csv        # Generated performance metrics
│   └── temp_images/                 # Temporary test images
│
└── docs/                             # Doxygen-generated documentation
    └── html/                         # HTML documentation
```

## Usage Examples

### Console Demo (10 Examples)
```bash
python console_launcher.py
# Select option 1
```
Demonstrates:
1. Image loading and operations
2. Format conversion (GRAYSCALE ↔ RGB)
3. Image resizing
4. Sequential monochrome conversion (all 3 methods)
5. Sequential hue adjustment (batch and individual)
6. Parallel monochrome conversion
7. Parallel varying hue shifts
8. Batch hue adjustment
9-10. Performance comparisons

### GUI Application
```bash
python gui_launcher.py
```
Features:
- **Monochrome Tab**: Select conversion method, load image, process and save
- **Hue Tab**: Apply hue shift (-180° to +180°), batch processing support
- Thread-safe background processing with real-time updates

### Programmatic Usage
```python
from src.image import OpenCVImage
from src.modules import MonochromeConverterPar, HueAdjusterPar

# Load image
image = OpenCVImage("path/to/image.jpg")

# Sequential monochrome conversion
converter = MonochromeConverterSeq(method="standard")
mono_images = converter.process([image])

# Parallel hue adjustment
adjuster = HueAdjusterPar(num_workers=4)
shifted = adjuster.process_batch([image], [30])  # 30° shift
```

## Testing

### Run All Tests (217 tests)
```bash
cd tests
python run_all_tests.py
```

### Expected Result
```
Ran 217 tests in ~15 seconds
OK
```

### Test Coverage by Category
- Image I/O and format conversion
- Monochrome conversion correctness (all 3 methods)
- Hue adjustment accuracy
- Batch processing consistency
- Sequential vs Parallel comparison
- Thread safety validation
- GUI signal/slot handling
- Edge cases and error handling

## Performance Benchmarking

### Run Benchmarks
```bash
python benchmarks/benchmark.py
```

### What Gets Benchmarked
- Monochrome conversion: 3 methods × 3 thread counts
- Hue adjustment: Multiple shifts (30°, 60°, 90°) × 4 thread variants
- CSV output: `benchmarks/benchmark_results.csv`

### Example Results
- Monochrome standard (4 threads): ~2.3x speedup
- Hue adjustment 60° (4 threads): ~6.77x speedup
- Varies based on system hardware

## Architecture

### Design Patterns
- **Factory Pattern**: ProcessorFactory for processor instantiation
- **Abstract Base Class**: ProcessorBase for common processor interface
- **Multiprocessing**: ProcessPoolExecutor for true parallelism
- **Signal/Slot**: PyQt5 thread-safe communication

### Threading Model
- **Parallel Implementation**: ProcessPoolExecutor (multiprocessing)
- **GUI Threading**: QThread + signal/slot architecture
- **Thread-Safe Logger**: Mutex-protected logging
- **Worker Functions**: Module-level (required for pickling)

### Data Flow
```
Image Input → Format Validation → Processing → Output Image
                                    ↓
                          Sequential/Parallel
                          (Method dispatch)
                                    ↓
                          Color space operations
                          (RGB ↔ HSV)
```

## Documentation

### Doxygen Documentation
Generate and view API documentation:
```bash
# Generate (requires Doxygen)
doxygen Doxyfile

# View HTML documentation
start docs/html/index.html
```

### Documentation Includes
- 17 documented source files
- 20+ classes with inheritance diagrams
- 100+ functions with signatures
- Cross-references and source code links
- Search functionality

### Key Files
- `DOCUMENTATION.md` - Comprehensive guide (400+ lines)
- `DOC_SUMMARY.md` - Quick reference

## Code Quality

### Standards
- Python PEP 8 compliant
- Type hints where applicable
- Comprehensive docstrings (Doxygen format)
- Error handling and validation
- Edge case coverage

### Testing Standards
- 217 comprehensive unit tests
- All tests passing
- CI/CD ready
- Thread safety validated