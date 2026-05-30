import os
import sys
import time
import csv
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.image import OpenCVImage
from src.modules.sequential import MonochromeConverterSeq, HueAdjusterSeq
from src.modules.parallel import MonochromeConverterPar, HueAdjusterPar


class BenchmarkRunner:
    
    def __init__(self):
        self.results: Dict[str, List[Dict]] = {}
        self.benchmark_dir = Path(__file__).parent
    
    def create_sample_images(self, num_images: int, width: int, height: int) -> List[str]:
        temp_dir = self.benchmark_dir / "temp_images"
        temp_dir.mkdir(exist_ok=True)
        
        paths = []
        for i in range(num_images):
            path = temp_dir / f"sample_{i}.png"
            img_data = np.zeros((height, width, 3), dtype=np.uint8)
            img_data[0:height//3, :] = [0, 0, 255]
            img_data[height//3:2*height//3, :] = [0, 255, 0]
            img_data[2*height//3:height, :] = [255, 0, 0]
            
            cv2.imwrite(str(path), img_data)
            paths.append(str(path))
        
        return paths
    
    def benchmark_monochrome_conversion(self, image_paths: List[str], 
                                        methods: List[str] = None,
                                        num_threads_list: List[int] = None) -> List[Dict]:
        methods = methods or ['standard', 'luminosity', 'average']
        num_threads_list = num_threads_list or [1, 2, 4]
        
        print(f"Benchmarking monochrome conversion with {len(image_paths)} images")
        results = []
        
        images = []
        for path in image_paths:
            img = OpenCVImage()
            img.load(path)
            images.append(img)
            
        for method in methods:
            print(f"\n--- Method: {method} ---")
            seq_converter = MonochromeConverterSeq(method=method)
            start_time = time.time()
            for img in images:
                seq_converter.process(img)
            seq_time = time.time() - start_time
            
            result = {
                'test_case': 'Monochrome Conversion Sequential',
                'method': method,
                'num_images': len(images),
                'num_threads': 1,
                'execution_time': seq_time,
                'time_per_image': seq_time / len(images),
                'throughput': len(images) / seq_time
            }
            results.append(result)
            print(f"Sequential: {seq_time:.3f}s ({seq_time/len(images):.3f}s per image)")
            
            for num_threads in num_threads_list:
                par_converter = MonochromeConverterPar(method=method, num_threads=num_threads)
                start_time = time.time()
                par_converter.process_batch(images)
                par_time = time.time() - start_time
                
                result = {
                    'test_case': 'Monochrome Conversion Parallel',
                    'method': method,
                    'num_images': len(images),
                    'num_threads': num_threads,
                    'execution_time': par_time,
                    'time_per_image': par_time / len(images),
                    'throughput': len(images) / par_time,
                    'speedup': seq_time / par_time if par_time > 0 else 0
                }
                results.append(result)
                print(
                    f"Parallel ({num_threads} threads): {par_time:.3f}s "
                    f"(speedup: {seq_time/par_time:.2f}x)"
                )
        
        return results
    
    def benchmark_hue_adjustment(self, image_paths: List[str], 
                                hue_shifts: List[int] = None,
                                num_threads_list: List[int] = None) -> List[Dict]:
        hue_shifts = hue_shifts or [30, 60, 90]
        num_threads_list = num_threads_list or [1, 2, 4]
        
        print(f"Benchmarking hue adjustment with {len(image_paths)} images")
        results = []
        
        images = []
        for path in image_paths:
            img = OpenCVImage()
            img.load(path)
            images.append(img)
            
        for hue_shift in hue_shifts:
            print(f"\n--- Hue Shift: {hue_shift}° ---")
            
            start_time = time.time()
            for img in images:
                adjuster = HueAdjusterSeq(hue_shift=hue_shift)
                adjuster.process(img)
            seq_time = time.time() - start_time
            
            result = {
                'test_case': 'Hue Adjustment Sequential',
                'hue_shift': hue_shift,
                'num_images': len(images),
                'num_threads': 1,
                'execution_time': seq_time,
                'time_per_image': seq_time / len(images),
                'throughput': len(images) / seq_time
            }
            results.append(result)
            print(f"Sequential: {seq_time:.3f}s ({seq_time/len(images):.3f}s per image)")
            
            for num_threads in num_threads_list:
                adjuster_par = HueAdjusterPar(num_threads=num_threads)
                start_time = time.time()
                adjuster_par.adjust_hue_batch(images, hue_shift)
                par_time = time.time() - start_time
                
                result = {
                    'test_case': 'Hue Adjustment Parallel',
                    'hue_shift': hue_shift,
                    'num_images': len(images),
                    'num_threads': num_threads,
                    'execution_time': par_time,
                    'time_per_image': par_time / len(images),
                    'throughput': len(images) / par_time,
                    'speedup': seq_time / par_time if par_time > 0 else 0
                }
                results.append(result)
                print(
                    f"Parallel ({num_threads} threads): {par_time:.3f}s "
                    f"(speedup: {seq_time/par_time:.2f}x)"
                )
        
        return results
        
    def save_results(self, all_results: List[Dict], filename: str = "benchmark_results.csv"):
        output_file = self.benchmark_dir / filename
        
        if not all_results:
            return
        
        keys = set()
        for result in all_results:
            keys.update(result.keys())
        
        keys = sorted(list(keys))
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"Results saved to {output_file}")
    
    def run_all_benchmarks(self):
        all_results = []
        
        for num_images in [3, 5, 10]:
            print(f"\n{'='*60}")
            print(f"Benchmarking with {num_images} images")
            print(f"{'='*60}")
            
            print("Running benchmarks with sample images (400x400)")
            image_paths = self.create_sample_images(num_images, 400, 400)
            
            results = self.benchmark_monochrome_conversion(image_paths)
            all_results.extend(results)
            
            results = self.benchmark_hue_adjustment(image_paths)
            all_results.extend(results)
            
            for path in image_paths:
                os.remove(path)
        
        print(f"\n{'='*60}")
        print(f"Saving {len(all_results)} results")
        print(f"{'='*60}")
        self.save_results(all_results)
        
        return all_results


def main():
    runner = BenchmarkRunner()
    results = runner.run_all_benchmarks()
    
    print(f"\n{'='*60}")
    print(f"Benchmark completed! {len(results)} tests executed.")
    print(f"Results saved to: {Path(__file__).parent / 'benchmark_results.csv'}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
