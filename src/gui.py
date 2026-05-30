##\file gui.py
##\brief PyQt5 GUI application for image processing
##\details Provides graphical interface for monochrome conversion and hue adjustment
##\author Lab Team
##\version 1.0

import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QSlider, QTabWidget,
    QFileDialog, QProgressBar, QTextEdit, QMessageBox, QSpinBox
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

sys.path.insert(0, str(Path(__file__).parent))

from src.common import ImageFormat
from src.image import OpenCVImage
from src.modules import ProcessorFactory, ProcessingMode


##\brief Worker thread for image processing operations
##\details Base class for background image processing tasks
class ProcessingThread(QThread):
    ##\brief Signal emitted when processing is finished
    finished = pyqtSignal()
    ##\brief Signal emitted when result is ready, carries processed image
    result_ready = pyqtSignal(object)
    ##\brief Signal emitted when error occurs, carries error message
    error_occurred = pyqtSignal(str)


##\brief Thread for monochrome conversion processing
##\details Runs monochrome conversion in background thread
class MonochromeThread(ProcessingThread):
    ##\brief Initialize monochrome processing thread
    ##\param image OpenCVImage to process
    ##\param method Conversion method ("standard", "luminosity", "average")
    ##\param use_parallel Use parallel processing if True
    ##\param num_threads Number of threads for parallel processing
    def __init__(self, image: OpenCVImage, method: str, use_parallel: bool, num_threads: int = 4):
        super().__init__()
        self.image = image
        self.method = method
        self.use_parallel = use_parallel
        self.num_threads = num_threads
    
    def run(self):
        try:
            mode = ProcessingMode.PARALLEL if self.use_parallel else ProcessingMode.SEQUENTIAL
            converter = ProcessorFactory.create_monochrome_converter(
                mode=mode,
                method=self.method,
                num_threads=self.num_threads
            )
            result = converter.process(self.image)
            
            self.result_ready.emit(result)
        except Exception as e:
            self.error_occurred.emit(f"Error in monochrome conversion: {str(e)}")
        finally:
            self.finished.emit()


##\brief Thread for hue adjustment processing
##\details Runs hue adjustment in background thread
class HueThread(ProcessingThread):
    ##\brief Initialize hue adjustment thread
    ##\param image OpenCVImage to process
    ##\param hue_shift Hue shift value in degrees (-180 to 180)
    ##\param use_parallel Use parallel processing if True
    ##\param num_threads Number of threads for parallel processing
    def __init__(self, image: OpenCVImage, hue_shift: int, use_parallel: bool, num_threads: int = 4):
        super().__init__()
        self.image = image
        self.hue_shift = hue_shift
        self.use_parallel = use_parallel
        self.num_threads = num_threads
    
    def run(self):
        try:
            mode = ProcessingMode.PARALLEL if self.use_parallel else ProcessingMode.SEQUENTIAL
            adjuster = ProcessorFactory.create_hue_adjuster(
                mode=mode,
                num_threads=self.num_threads
            )
            
            if hasattr(adjuster, 'set_hue_shift'):
                adjuster.set_hue_shift(self.hue_shift)
            
            result = adjuster.process(self.image)
            
            self.result_ready.emit(result)
        except Exception as e:
            self.error_occurred.emit(f"Error in hue adjustment: {str(e)}")
        finally:
            self.finished.emit()


##\brief Thread for performance comparison
##\details Runs performance comparison tests in background
class ComparisonThread(ProcessingThread):
    ##\brief Initialize comparison thread
    ##\param image OpenCVImage for testing
    ##\param operation "monochrome" or "hue"
    ##\param method_or_shift Conversion method or hue shift value
    ##\param num_runs Number of test runs
    def __init__(self, image: OpenCVImage, operation: str, method_or_shift, num_runs: int = 3):
        super().__init__()
        self.image = image
        self.operation = operation
        self.method_or_shift = method_or_shift
        self.num_runs = num_runs
        self.results = {}
    
    def run(self):
        try:
            if self.operation == 'monochrome':
                self.compare_monochrome()
            else:
                self.compare_hue()
            self.result_ready.emit(self.results)
        except Exception as e:
            self.error_occurred.emit(f"Error in comparison: {str(e)}")
        finally:
            self.finished.emit()
    
    def compare_monochrome(self):
        method = self.method_or_shift
        
        seq_times = []
        for _ in range(self.num_runs):
            start = time.time()
            converter = ProcessorFactory.create_monochrome_converter(
                mode=ProcessingMode.SEQUENTIAL,
                method=method
            )
            converter.process(self.image)
            seq_times.append(time.time() - start)
        
        par_times = {}
        for num_threads in [1, 2, 4]:
            times = []
            for _ in range(self.num_runs):
                start = time.time()
                converter = ProcessorFactory.create_monochrome_converter(
                    mode=ProcessingMode.PARALLEL,
                    method=method,
                    num_threads=num_threads
                )
                converter.process(self.image)
                times.append(time.time() - start)
            par_times[num_threads] = times
        
        self.results = {
            'seq_avg': sum(seq_times) / len(seq_times),
            'seq_times': seq_times,
            'par_times': par_times
        }
    
    def compare_hue(self):
        hue_shift = self.method_or_shift
        
        seq_times = []
        for _ in range(self.num_runs):
            start = time.time()
            adjuster = ProcessorFactory.create_hue_adjuster(
                mode=ProcessingMode.SEQUENTIAL
            )
            adjuster.set_hue_shift(hue_shift)
            adjuster.process(self.image)
            seq_times.append(time.time() - start)
        
        par_times = {}
        for num_threads in [1, 2, 4]:
            times = []
            for _ in range(self.num_runs):
                start = time.time()
                adjuster = ProcessorFactory.create_hue_adjuster(
                    mode=ProcessingMode.PARALLEL,
                    num_threads=num_threads
                )
                adjuster.adjust_hue_batch([self.image], hue_shift)
                times.append(time.time() - start)
            par_times[num_threads] = times
        
        self.results = {
            'seq_avg': sum(seq_times) / len(seq_times),
            'seq_times': seq_times,
            'par_times': par_times
        }


##\brief Image viewer widget
##\details Displays OpenCV images using PyQt5
class ImageViewer(QLabel):
    ##\brief Initialize image viewer
    def __init__(self):
        super().__init__()
        self.setMinimumSize(500, 400)
        self.setStyleSheet("border: 2px solid #ccc; background-color: #f5f5f5;")
        self.setAlignment(Qt.AlignCenter)
        self.setText("No image loaded")
        self.setScaledContents(False)
    
    def display(self, cv_image):
        ##\brief Display OpenCV image in widget
        ##\param cv_image OpenCV image as numpy array
        if cv_image is None or cv_image.size == 0:
            self.setText("Invalid image")
            return
        
        try:
            rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            h, w = rgb.shape[:2]
            max_dim = 500
            if h > max_dim or w > max_dim:
                scale = min(max_dim / h, max_dim / w)
                rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))
            
            h, w, ch = rgb.shape
            bytes_per_line = 3 * w
            qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_img)
            
            self.setPixmap(pixmap)
        except Exception as e:
            self.setText(f"Error displaying image: {str(e)}")


##\brief Tab for monochrome conversion
##\details UI tab for monochrome color conversion operations
class MonochromeTab(QWidget):
    ##\brief Initialize monochrome tab
    def __init__(self):
        super().__init__()
        self.current_image: Optional[OpenCVImage] = None
        self.processed_image: Optional[OpenCVImage] = None
        self.current_thread: Optional[ProcessingThread] = None
        self.init_ui()
    
    def init_ui(self):
        main_layout = QHBoxLayout()
        
        left_layout = QVBoxLayout()
        self.image_viewer = ImageViewer()
        left_layout.addWidget(self.image_viewer)
        main_layout.addLayout(left_layout, 2)
        
        right_layout = QVBoxLayout()
        
        right_layout.addWidget(QLabel("<b>Image</b>", font=QFont("Arial", 10)))
        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self.load_image)
        right_layout.addWidget(load_btn)
        
        right_layout.addSpacing(20)
        
        right_layout.addWidget(QLabel("<b>Conversion Method</b>", font=QFont("Arial", 10)))
        self.method_combo = QComboBox()
        self.method_combo.addItems(['standard', 'luminosity', 'average'])
        right_layout.addWidget(self.method_combo)
        
        right_layout.addSpacing(20)
        
        right_layout.addWidget(QLabel("<b>Process</b>", font=QFont("Arial", 10)))
        
        seq_btn = QPushButton("Sequential")
        seq_btn.clicked.connect(self.process_sequential)
        right_layout.addWidget(seq_btn)
        
        par_btn = QPushButton("Parallel (4 threads)")
        par_btn.clicked.connect(self.process_parallel)
        right_layout.addWidget(par_btn)
        
        cmp_btn = QPushButton("Compare Performance")
        cmp_btn.clicked.connect(self.compare)
        right_layout.addWidget(cmp_btn)
        
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        right_layout.addWidget(self.progress)
        
        right_layout.addSpacing(15)
        
        export_btn = QPushButton("Export Result")
        export_btn.clicked.connect(self.export_image)
        right_layout.addWidget(export_btn)
        
        right_layout.addSpacing(15)
        
        right_layout.addWidget(QLabel("<b>Results</b>", font=QFont("Arial", 10)))
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(200)
        right_layout.addWidget(self.results_text)
        
        right_layout.addStretch()
        main_layout.addLayout(right_layout, 1)
        
        self.setLayout(main_layout)
    
    def load_image(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )
        
        if filepath:
            try:
                self.current_image = OpenCVImage()
                if self.current_image.load(filepath):
                    self.image_viewer.display(self.current_image.data)
                    self.results_text.setText(
                        f"✓ Loaded: {Path(filepath).name}\n"
                        f"  Size: {self.current_image.width}×{self.current_image.height}"
                    )
                else:
                    QMessageBox.warning(self, "Error", "Failed to load image")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading image: {str(e)}")
    
    def process_sequential(self):
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "No image loaded")
            return
        
        method = self.method_combo.currentText()
        self.results_text.setText(f"Processing with {method}...\n")
        self.progress.setVisible(True)
        self.progress.setValue(0)
        
        self.current_thread = MonochromeThread(self.current_image, method, False)
        self.current_thread.result_ready.connect(self.on_result)
        self.current_thread.error_occurred.connect(self.on_error)
        self.current_thread.finished.connect(lambda: self.progress.setVisible(False))
        self.current_thread.start()
    
    def process_parallel(self):
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "No image loaded")
            return
        
        method = self.method_combo.currentText()
        self.results_text.setText(f"Processing with {method} (parallel)...\n")
        self.progress.setVisible(True)
        self.progress.setValue(0)
        
        self.current_thread = MonochromeThread(self.current_image, method, True, num_threads=4)
        self.current_thread.result_ready.connect(self.on_result)
        self.current_thread.error_occurred.connect(self.on_error)
        self.current_thread.finished.connect(lambda: self.progress.setVisible(False))
        self.current_thread.start()
    
    def compare(self):
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "No image loaded")
            return
        
        method = self.method_combo.currentText()
        self.results_text.setText(f"Comparing performance for {method}...\n")
        self.progress.setVisible(True)
        self.progress.setValue(0)
        
        self.current_thread = ComparisonThread(self.current_image, 'monochrome', method)
        self.current_thread.result_ready.connect(self.on_comparison_result)
        self.current_thread.error_occurred.connect(self.on_error)
        self.current_thread.finished.connect(lambda: self.progress.setVisible(False))
        self.current_thread.start()
    
    def on_result(self, image: OpenCVImage):
        if image:
            self.processed_image = image
            self.image_viewer.display(image.data)
            self.results_text.append("✓ Conversion complete")
            self.progress.setValue(100)
    
    def on_comparison_result(self, results: dict):
        seq_avg = results.get('seq_avg', 0)
        par_times = results.get('par_times', {})
        
        text = f"Sequential: {seq_avg*1000:.2f}ms\n"
        for threads, times in par_times.items():
            avg = sum(times) / len(times)
            speedup = seq_avg / avg if avg > 0 else 0
            text += f"Parallel ({threads}T): {avg*1000:.2f}ms (×{speedup:.2f})\n"
        
        self.results_text.append(text)
        self.progress.setValue(100)
    
    def on_error(self, error_msg: str):
        self.results_text.append(f"✗ {error_msg}")
        self.progress.setVisible(False)
    
    def export_image(self):
        if self.processed_image is None or self.processed_image.data is None:
            QMessageBox.warning(self, "Warning", "No processed image to export")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Image", "",
            "PNG Image (*.png);;JPEG Image (*.jpg);;BMP Image (*.bmp);;All Files (*)"
        )
        
        if filepath:
            try:
                if not cv2.imwrite(filepath, self.processed_image.data):
                    QMessageBox.warning(self, "Error", "Failed to save image")
                else:
                    QMessageBox.information(self, "Success", f"Image exported to:\n{filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error exporting image: {str(e)}")


##\brief Tab for hue adjustment
##\details UI tab for hue color adjustment operations
class HueTab(QWidget):
    ##\brief Initialize hue adjustment tab
    def __init__(self):
        super().__init__()
        self.current_image: Optional[OpenCVImage] = None
        self.processed_image: Optional[OpenCVImage] = None
        self.current_thread: Optional[ProcessingThread] = None
        self.init_ui()
    
    def init_ui(self):
        main_layout = QHBoxLayout()
        
        left_layout = QVBoxLayout()
        self.image_viewer = ImageViewer()
        left_layout.addWidget(self.image_viewer)
        main_layout.addLayout(left_layout, 2)
        
        right_layout = QVBoxLayout()
        
        right_layout.addWidget(QLabel("<b>Image</b>", font=QFont("Arial", 10)))
        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self.load_image)
        right_layout.addWidget(load_btn)
        
        right_layout.addSpacing(20)
        
        right_layout.addWidget(QLabel("<b>Hue Shift</b>", font=QFont("Arial", 10)))
        
        shift_layout = QHBoxLayout()
        shift_layout.addWidget(QLabel("-180°"))
        
        self.hue_slider = QSlider(Qt.Horizontal)
        self.hue_slider.setRange(-180, 180)
        self.hue_slider.setValue(0)
        self.hue_slider.setTickPosition(QSlider.TicksBelow)
        self.hue_slider.setTickInterval(30)
        shift_layout.addWidget(self.hue_slider)
        
        shift_layout.addWidget(QLabel("180°"))
        right_layout.addLayout(shift_layout)
        
        self.hue_value = QSpinBox()
        self.hue_value.setRange(-180, 180)
        self.hue_value.setValue(0)
        self.hue_slider.valueChanged.connect(self.hue_value.setValue)
        self.hue_value.valueChanged.connect(self.hue_slider.setValue)
        right_layout.addWidget(self.hue_value)
        
        right_layout.addSpacing(20)
        
        right_layout.addWidget(QLabel("<b>Process</b>", font=QFont("Arial", 10)))
        
        seq_btn = QPushButton("Sequential")
        seq_btn.clicked.connect(self.process_sequential)
        right_layout.addWidget(seq_btn)
        
        par_btn = QPushButton("Parallel (4 threads)")
        par_btn.clicked.connect(self.process_parallel)
        right_layout.addWidget(par_btn)
        
        cmp_btn = QPushButton("Compare Performance")
        cmp_btn.clicked.connect(self.compare)
        right_layout.addWidget(cmp_btn)
        
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        right_layout.addWidget(self.progress)
        
        right_layout.addSpacing(15)
        
        export_btn = QPushButton("Export Result")
        export_btn.clicked.connect(self.export_image)
        right_layout.addWidget(export_btn)
        
        right_layout.addSpacing(15)
        
        right_layout.addWidget(QLabel("<b>Results</b>", font=QFont("Arial", 10)))
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(200)
        right_layout.addWidget(self.results_text)
        
        right_layout.addStretch()
        main_layout.addLayout(right_layout, 1)
        
        self.setLayout(main_layout)
    
    def load_image(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )
        
        if filepath:
            try:
                self.current_image = OpenCVImage()
                if self.current_image.load(filepath):
                    self.image_viewer.display(self.current_image.data)
                    self.results_text.setText(
                        f"✓ Loaded: {Path(filepath).name}\n"
                        f"  Size: {self.current_image.width}×{self.current_image.height}"
                    )
                else:
                    QMessageBox.warning(self, "Error", "Failed to load image")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading image: {str(e)}")
    
    def process_sequential(self):
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "No image loaded")
            return
        
        hue_shift = self.hue_value.value()
        self.results_text.setText(f"Adjusting hue by {hue_shift}° (sequential)...\n")
        self.progress.setVisible(True)
        self.progress.setValue(0)
        
        self.current_thread = HueThread(self.current_image, hue_shift, False)
        self.current_thread.result_ready.connect(self.on_result)
        self.current_thread.error_occurred.connect(self.on_error)
        self.current_thread.finished.connect(lambda: self.progress.setVisible(False))
        self.current_thread.start()
    
    def process_parallel(self):
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "No image loaded")
            return
        
        hue_shift = self.hue_value.value()
        self.results_text.setText(f"Adjusting hue by {hue_shift}° (parallel)...\n")
        self.progress.setVisible(True)
        self.progress.setValue(0)
        
        self.current_thread = HueThread(self.current_image, hue_shift, True, num_threads=4)
        self.current_thread.result_ready.connect(self.on_result)
        self.current_thread.error_occurred.connect(self.on_error)
        self.current_thread.finished.connect(lambda: self.progress.setVisible(False))
        self.current_thread.start()
    
    def compare(self):
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "No image loaded")
            return
        
        hue_shift = self.hue_value.value()
        self.results_text.setText(f"Comparing performance for hue shift {hue_shift}°...\n")
        self.progress.setVisible(True)
        self.progress.setValue(0)
        
        self.current_thread = ComparisonThread(self.current_image, 'hue', hue_shift)
        self.current_thread.result_ready.connect(self.on_comparison_result)
        self.current_thread.error_occurred.connect(self.on_error)
        self.current_thread.finished.connect(lambda: self.progress.setVisible(False))
        self.current_thread.start()
    
    def on_result(self, image: OpenCVImage):
        if image:
            self.processed_image = image
            self.image_viewer.display(image.data)
            self.results_text.append("✓ Adjustment complete")
            self.progress.setValue(100)
    
    def on_comparison_result(self, results: dict):
        seq_avg = results.get('seq_avg', 0)
        par_times = results.get('par_times', {})
        
        text = f"Sequential: {seq_avg*1000:.2f}ms\n"
        for threads, times in par_times.items():
            avg = sum(times) / len(times)
            speedup = seq_avg / avg if avg > 0 else 0
            text += f"Parallel ({threads}T): {avg*1000:.2f}ms (×{speedup:.2f})\n"
        
        self.results_text.append(text)
        self.progress.setValue(100)
    
    def on_error(self, error_msg: str):
        self.results_text.append(f"✗ {error_msg}")
        self.progress.setVisible(False)
    
    def export_image(self):
        if self.processed_image is None or self.processed_image.data is None:
            QMessageBox.warning(self, "Warning", "No processed image to export")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Image", "",
            "PNG Image (*.png);;JPEG Image (*.jpg);;BMP Image (*.bmp);;All Files (*)"
        )
        
        if filepath:
            try:
                if not cv2.imwrite(filepath, self.processed_image.data):
                    QMessageBox.warning(self, "Error", "Failed to save image")
                else:
                    QMessageBox.information(self, "Success", f"Image exported to:\n{filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error exporting image: {str(e)}")


##\brief Main application window
##\details Contains tabbed interface for image processing operations
class MainWindow(QMainWindow):
    ##\brief Initialize main window
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("Image Color Processing - GUI")
        self.setGeometry(100, 100, 1200, 700)
        
        tabs = QTabWidget()
        tabs.addTab(MonochromeTab(), "Monochrome Conversion")
        tabs.addTab(HueTab(), "Hue Adjustment")
        
        self.setCentralWidget(tabs)
        self.statusBar().showMessage("Ready")
    
    def closeEvent(self, event):
        ##\brief Handle window close event
        ##\param event Close event
        event.accept()


def run_gui():
    ##\brief Run PyQt5 GUI application
    ##\details Creates and shows main window with event loop
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    run_gui()
