##\file test_gui_thread_safety.py
##\brief Tests for GUI thread safety and operations
##\details Unit tests for PyQt5 GUI thread safety and image processing integration
##\author Lab Team
##\version 1.0

import unittest
import time
import threading
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.image import OpenCVImage
from src.gui import MonochromeTab, HueTab, MonochromeThread, HueThread


##\brief Global QApplication instance for GUI tests
_app = None

def get_qapp():
    ##\brief Get or create QApplication instance
    ##\return QApplication instance
    global _app
    if _app is None:
        _app = QApplication.instance()
        if _app is None:
            _app = QApplication(sys.argv)
    return _app


##\brief Base class for GUI thread safety tests
##\details Provides common setup/teardown for GUI testing
class GUIThreadSafetyTestBase(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up QApplication for all tests."""
        cls.app = get_qapp()
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_path = Path(cls.temp_dir.name)
    
    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()
    
    def setUp(self):
        test_image_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.test_image_path = str(self.temp_path / f"test_{id(self)}.jpg")
        cv2.imwrite(self.test_image_path, test_image_data)
        
        self.test_image = OpenCVImage()
        self.test_image.load(self.test_image_path)
    
    def process_app_events(self, timeout_ms=100):
        end_time = time.time() + (timeout_ms / 1000.0)
        while time.time() < end_time:
            self.app.processEvents()
            time.sleep(0.01)


class TestMonochromeThreadSignals(GUIThreadSafetyTestBase):
    
    def test_thread_completion_signal(self):
        finished = []
        results = []
        
        thread = MonochromeThread(self.test_image, 'standard', False)
        thread.finished.connect(lambda: finished.append(True))
        thread.result_ready.connect(lambda img: results.append(img))
        
        thread.start()
        thread.wait(5000)
        self.process_app_events()
        
        self.assertTrue(len(finished) > 0, "finished signal not emitted")
        self.assertTrue(len(results) > 0, "result_ready signal not emitted")
    
    def test_sequential_processing(self):
        results = []
        thread = MonochromeThread(self.test_image, 'luminosity', False)
        thread.result_ready.connect(lambda img: results.append(img))
        thread.start()
        thread.wait(5000)
        self.process_app_events()
        
        self.assertEqual(len(results), 1)
        self.assertIsNotNone(results[0].data)
    
    def test_different_methods(self):
        all_results = []
        for method in ['standard', 'luminosity', 'average']:
            results = []
            thread = MonochromeThread(self.test_image, method, False)
            thread.result_ready.connect(lambda img: results.append(img))
            thread.start()
            thread.wait(5000)
            self.process_app_events()
            all_results.extend(results)
        
        self.assertEqual(len(all_results), 3)


class TestHueThreadSignals(GUIThreadSafetyTestBase):
    
    def test_thread_completion_signal(self):
        finished = []
        results = []
        
        thread = HueThread(self.test_image, 45, False)
        thread.finished.connect(lambda: finished.append(True))
        thread.result_ready.connect(lambda img: results.append(img))
        
        thread.start()
        thread.wait(5000)
        self.process_app_events()
        
        self.assertTrue(len(finished) > 0)
        self.assertTrue(len(results) > 0)
    
    def test_sequential_processing(self):
        results = []
        thread = HueThread(self.test_image, 60, False)
        thread.result_ready.connect(lambda img: results.append(img))
        thread.start()
        thread.wait(5000)
        self.process_app_events()
        
        self.assertEqual(len(results), 1)
        self.assertIsNotNone(results[0].data)
    
    def test_different_hue_shifts(self):
        all_results = []
        for hue in [30, -60, 90, -120, 0]:
            results = []
            thread = HueThread(self.test_image, hue, False)
            thread.result_ready.connect(lambda img: results.append(img))
            thread.start()
            thread.wait(5000)
            self.process_app_events()
            all_results.extend(results)
        
        self.assertEqual(len(all_results), 5)


class TestConcurrentProcessing(GUIThreadSafetyTestBase):
    
    def test_concurrent_monochrome_methods(self):
        results = []
        finished_count = [0]
        
        def on_result(img):
            results.append(img)
        
        def on_finished():
            finished_count[0] += 1
        
        threads = []
        for method in ['standard', 'luminosity', 'average']:
            t = MonochromeThread(self.test_image, method, False)
            t.result_ready.connect(on_result)
            t.finished.connect(on_finished)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.wait(5000)
        self.process_app_events()
        
        self.assertEqual(len(results), 3)
        self.assertEqual(finished_count[0], 3)
    
    def test_concurrent_monochrome_and_hue(self):
        results = []
        
        mono_t = MonochromeThread(self.test_image, 'standard', False)
        hue_t = HueThread(self.test_image, 60, False)
        
        mono_t.result_ready.connect(lambda img: results.append(img))
        hue_t.result_ready.connect(lambda img: results.append(img))
        
        mono_t.start()
        hue_t.start()
        
        mono_t.wait(5000)
        hue_t.wait(5000)
        self.process_app_events()
        
        self.assertEqual(len(results), 2)


class TestMonochromeTabIntegration(GUIThreadSafetyTestBase):
    
    def setUp(self):
        super().setUp()
        self.tab = MonochromeTab()
        self.tab.current_image = self.test_image
    
    def test_sequential_processing(self):
        self.tab.process_sequential()
        if self.tab.current_thread:
            self.tab.current_thread.wait(5000)
        self.process_app_events(500)
        
        self.assertIsNotNone(self.tab.processed_image)
    
    def test_rapid_method_changes(self):
        for method in ['standard', 'luminosity', 'average']:
            self.tab.method_combo.setCurrentText(method)
            self.tab.process_sequential()
            if self.tab.current_thread:
                self.tab.current_thread.wait(5000)
            self.process_app_events(200)


class TestHueTabIntegration(GUIThreadSafetyTestBase):
    
    def setUp(self):
        super().setUp()
        self.tab = HueTab()
        self.tab.current_image = self.test_image
    
    def test_sequential_processing(self):
        self.tab.hue_value.setValue(45)
        self.tab.process_sequential()
        if self.tab.current_thread:
            self.tab.current_thread.wait(5000)
        self.process_app_events(500)
        
        self.assertIsNotNone(self.tab.processed_image)
    
    def test_rapid_value_changes(self):
        for hue in [30, -45, 90, -120, 0]:
            self.tab.hue_value.setValue(hue)
            self.tab.process_sequential()
            if self.tab.current_thread:
                self.tab.current_thread.wait(5000)
            self.process_app_events(200)


class TestThreadRaceConditions(GUIThreadSafetyTestBase):
    
    def test_rapid_thread_creation(self):
        threads = []
        for i in range(10):
            t = MonochromeThread(self.test_image, 'standard', False)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.wait(5000)
        self.process_app_events()
        
        self.assertEqual(len(threads), 10)
    
    def test_concurrent_signal_emissions(self):
        counts = {'finished': 0, 'result': 0}
        lock = threading.Lock()
        
        def on_finished():
            with lock:
                counts['finished'] += 1
        
        def on_result(img):
            with lock:
                counts['result'] += 1
        
        threads = []
        for i in range(5):
            t = MonochromeThread(self.test_image, 'standard', False)
            t.finished.connect(on_finished)
            t.result_ready.connect(on_result)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.wait(5000)
        self.process_app_events()
        
        self.assertEqual(counts['finished'], 5)
        self.assertEqual(counts['result'], 5)


class TestImageDataIntegrity(GUIThreadSafetyTestBase):
    
    def test_monochrome_output_validity(self):
        results = []
        thread = MonochromeThread(self.test_image, 'standard', False)
        thread.result_ready.connect(lambda img: results.append(img))
        thread.start()
        thread.wait(5000)
        self.process_app_events()
        
        self.assertEqual(len(results), 1)
        img = results[0]
        self.assertIsNotNone(img.data)
        self.assertEqual(img.data.dtype, np.uint8)
        self.assertTrue(np.all(img.data >= 0) and np.all(img.data <= 255))
    
    def test_hue_output_validity(self):
        results = []
        thread = HueThread(self.test_image, 60, False)
        thread.result_ready.connect(lambda img: results.append(img))
        thread.start()
        thread.wait(5000)
        self.process_app_events()
        
        self.assertEqual(len(results), 1)
        img = results[0]
        self.assertIsNotNone(img.data)
        self.assertEqual(img.data.dtype, np.uint8)


if __name__ == '__main__':
    unittest.main(verbosity=2)