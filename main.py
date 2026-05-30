##\file main.py
##\brief Main application module with menu and console demo
##\details This module provides the main entry point for the Image Color Processing Application.
##It includes console menu navigation and demonstration of all image processing features.
##\author Lab Team
##\version 1.0
##\date 2025

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.common import ImageFormat
from src.image import OpenCVImage
from src.modules import (
    MonochromeConverterSeq, MonochromeConverterPar,
    HueAdjusterSeq, HueAdjusterPar
)


def show_menu():
    ##\brief Display main application menu
    ##\return User's menu choice as string
    print("\n" + "="*60)
    print("Image Color Processing Application")
    print("="*60)
    print("\nSelect mode:")
    print("  [1] Console Application")
    print("  [2] GUI Application")
    print("  [0] Exit")
    print("="*60)
    
    choice = input("\nEnter choice [0-2]: ").strip()
    return choice


def console_mode():
    ##\brief Start console application mode
    print("\nStarting Console Application...\n")
    run_console_demo()


def gui_mode():
    ##\brief Start GUI application mode
    print("\nStarting GUI Application...\n")
    print("Opening graphical interface...")
    from src.gui import run_gui
    run_gui()


def run_console_demo():
    ##\brief Run console demonstration of all image processing features
    ##\details Demonstrates loading images, format conversion, resizing,
    ##sequential and parallel monochrome conversion, hue adjustment, and performance comparison
    print("="*60)
    print("Image Color Processing Application - Example Usage")
    print("="*60)
    
    print("\n--- Example 1: Image Loading and Basic Operations ---")
    
    img = OpenCVImage()
    print("Created OpenCVImage instance")
    
    import numpy as np
    import cv2
    
    sample_data = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    sample_data[100:300, 100:300] = [0, 255, 0]
    sample_data[50:150, 50:150] = [255, 0, 0]
    sample_data[200:300, 200:300] = [0, 0, 255]
    
    sample_path = "sample_image.png"
    cv2.imwrite(sample_path, sample_data)
    
    success = img.load(sample_path)
    print(f"Loaded image: {success}, Dimensions: {img.width}x{img.height}")
    
    print("\n--- Example 2: Image Format Conversion ---")
    
    gray = img.convert_format(ImageFormat.GRAYSCALE)
    print(f"Converted to GRAYSCALE: {gray.width}x{gray.height}")
    
    rgb = img.convert_format(ImageFormat.RGB)
    print(f"Converted to RGB: {rgb.width}x{rgb.height}")
    
    print("\n--- Example 3: Image Resizing ---")
    
    small = img.resize(200, 200)
    print(f"Resized to 200x200: {small.width}x{small.height}")
    
    print("\n--- Example 4: Sequential Monochrome Conversion ---")
    
    converter_seq = MonochromeConverterSeq(method='standard')
    result_seq = converter_seq.process(img)
    print(f"Converted image to monochrome (method: standard)")
    print(f"Result dimensions: {result_seq.width}x{result_seq.height}")
    
    for method in ['luminosity', 'average']:
        converter = MonochromeConverterSeq(method=method)
        result = converter.process(img)
        print(f"Method '{method}' conversion completed")
    
    print("\n--- Example 5: Sequential Hue Adjustment ---")
    
    adjuster_seq = HueAdjusterSeq(hue_shift=30)
    result_hue = adjuster_seq.process(img)
    print(f"Adjusted hue by 30 degrees")
    print(f"Result dimensions: {result_hue.width}x{result_hue.height}")
    
    hue_values = [-90, -45, 0, 45, 90]
    for hue in hue_values:
        adjuster = HueAdjusterSeq(hue_shift=hue)
        result = adjuster.process(img)
        print(f"Hue shift {hue}° applied")
    
    print("\n--- Example 6: Parallel Monochrome Conversion ---")
    
    images = [img.clone() for _ in range(3)]
    converter_par = MonochromeConverterPar(method='standard', num_threads=2)
    
    start = time.time()
    results = converter_par.process_batch(images)
    elapsed = time.time() - start
    
    print(f"Processed {len(images)} images in {elapsed:.3f}s")
    print(f"All images converted to monochrome")
    
    print("\n--- Example 7: Parallel Hue Adjustment ---")
    
    adjuster_par = HueAdjusterPar(num_threads=2)
    hue_shifts = [45, 90, 135]
    
    start = time.time()
    adjusted = adjuster_par.adjust_hue_varying([img.clone() for _ in hue_shifts], hue_shifts)
    elapsed = time.time() - start
    
    print(f"Applied varying hue shifts in {elapsed:.3f}s")
    for i, hue in enumerate(hue_shifts):
        print(f"  Image {i}: Hue shifted by {hue}°")
    
    print("\n--- Example 8: Batch Hue Adjustment (Same Value) ---")
    
    sample_images = [img.clone() for _ in range(4)]
    
    start = time.time()
    adjusted_batch = adjuster_par.adjust_hue_batch(sample_images, 60)
    elapsed = time.time() - start
    
    print(f"Batch adjusted {len(sample_images)} images by 60° in {elapsed:.3f}s")
    
    print("\n--- Example 9: Performance Comparison (Monochrome) ---")
    
    sample_images_seq = [img.clone() for _ in range(5)]
    
    start = time.time()
    for sample_img in sample_images_seq:
        conv_seq = MonochromeConverterSeq(method='standard')
        conv_seq.process(sample_img)
    seq_time = time.time() - start
    
    conv_par = MonochromeConverterPar(method='standard', num_threads=4)
    start = time.time()
    conv_par.process_batch(sample_images_seq)
    par_time = time.time() - start
    
    print(f"Sequential processing (5 images): {seq_time:.3f}s")
    print(f"Parallel processing (5 images): {par_time:.3f}s")
    speedup = seq_time/par_time if par_time > 0 else 0
    print(f"Speedup: {speedup:.2f}x")
    
    print("\n--- Example 10: Performance Comparison (Hue Adjustment) ---")
    
    sample_images_hue = [img.clone() for _ in range(5)]
    
    start = time.time()
    for sample_img in sample_images_hue:
        adj_seq = HueAdjusterSeq(hue_shift=45)
        adj_seq.process(sample_img)
    seq_time_hue = time.time() - start
    
    adj_par = HueAdjusterPar(num_threads=4)
    start = time.time()
    adj_par.adjust_hue_batch(sample_images_hue, 45)
    par_time_hue = time.time() - start
    
    print(f"Sequential hue adjustment (5 images): {seq_time_hue:.3f}s")
    print(f"Parallel hue adjustment (5 images): {par_time_hue:.3f}s")
    speedup_hue = seq_time_hue/par_time_hue if par_time_hue > 0 else 0
    print(f"Speedup: {speedup_hue:.2f}x")
    
    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)


def main():
    ##\brief Main application entry point
    ##\details Displays menu and launches selected application mode
    while True:
        choice = show_menu()
        
        if choice == '1':
            console_mode()
            break
        elif choice == '2':
            gui_mode()
            break
        elif choice == '0':
            print("\nExiting...")
            sys.exit(0)
        else:
            print("Invalid choice. Please try again.")


if __name__ == '__main__':
    main()