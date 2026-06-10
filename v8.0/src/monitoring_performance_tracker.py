#!/usr/bin/env python3
"""
Performance Monitoring System for Chess Engine Neural Network Training

Tracks GPU, CPU, memory, and training metrics in real-time to prevent
system overload and understand compute requirements for scaling.

Dependencies:
  pip install nvidia-ml-py3 psutil numpy
"""

import time
import json
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import psutil
except ImportError:
    print("⚠️  psutil not installed. Run: pip install psutil")
    psutil = None

try:
    import pynvml
    pynvml.nvmlInit()
    NVIDIA_AVAILABLE = True
except (ImportError, Exception):
    print("⚠️  nvidia-ml-py3 not installed or NVIDIA GPU not available. Run: pip install nvidia-ml-py3")
    NVIDIA_AVAILABLE = False


class PerformanceMonitor:
    """Real-time monitoring of GPU, CPU, and training metrics"""
    
    # Alert thresholds
    TEMP_WARNING = 75  # Celsius
    TEMP_CRITICAL = 85
    GPU_MEM_WARNING = 0.9  # 90% usage
    CPU_TEMP_WARNING = 80
    
    def __init__(self, model_name: str = "v7p3r_nn_phase1", log_dir: Optional[Path] = None):
        """
        Initialize performance monitor
        
        Args:
            model_name: Name of model being trained
            log_dir: Directory to save metrics (default: ./monitoring/logs/)
        """
        self.model_name = model_name
        self.log_dir = Path(log_dir or "./monitoring/logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.metrics = defaultdict(list)
        self.start_time = time.time()
        self.gpu_handle = None
        
        # Setup logging
        self._setup_logging()
        
        # Initialize GPU if available
        if NVIDIA_AVAILABLE:
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_name = pynvml.nvmlDeviceGetName(self.gpu_handle).decode()
                self.logger.info(f"✅ NVIDIA GPU detected: {gpu_name}")
            except Exception as e:
                self.logger.warning(f"⚠️  Could not initialize GPU monitoring: {e}")
                self.gpu_handle = None
        
        # Check CPU temperature sensor availability
        self.has_cpu_temp = psutil and hasattr(psutil, 'sensors_temperatures')
    
    def _setup_logging(self):
        """Setup logging to file and console"""
        self.logger = logging.getLogger("PerformanceMonitor")
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        fh = logging.FileHandler(self.log_dir / f"{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def track_gpu(self) -> Dict[str, float]:
        """
        Track GPU utilization, memory, temperature, power
        
        Returns:
            Dictionary with GPU metrics
        """
        if not NVIDIA_AVAILABLE or not self.gpu_handle:
            return {}
        
        try:
            # Utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            
            # Memory
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            mem_used_gb = mem.used / 1e9
            mem_total_gb = mem.total / 1e9
            mem_percent = (mem.used / mem.total) * 100
            
            # Temperature
            temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, 0)
            
            # Power (in milliwatts → watts)
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle)
                power_w = power_mw / 1000
            except:
                power_w = 0
            
            # Get power limit
            try:
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(self.gpu_handle) / 1000
            except:
                power_limit = 0
            
            metrics = {
                'gpu_util_percent': util.gpu,
                'gpu_mem_percent': mem_percent,
                'gpu_mem_used_gb': mem_used_gb,
                'gpu_mem_total_gb': mem_total_gb,
                'gpu_temp_c': temp,
                'gpu_power_w': power_w,
                'gpu_power_limit_w': power_limit,
            }
            
            # Alert if thresholds exceeded
            if temp >= self.TEMP_CRITICAL:
                self.logger.error(f"🔴 CRITICAL: GPU temperature {temp}°C exceeds {self.TEMP_CRITICAL}°C!")
            elif temp >= self.TEMP_WARNING:
                self.logger.warning(f"🟡 WARNING: GPU temperature {temp}°C exceeds {self.TEMP_WARNING}°C")
            
            if mem_percent >= self.GPU_MEM_WARNING * 100:
                self.logger.warning(f"🟡 WARNING: GPU memory {mem_percent:.1f}% full")
            
            return metrics
        
        except Exception as e:
            self.logger.debug(f"Error reading GPU metrics: {e}")
            return {}
    
    def track_cpu(self) -> Dict[str, float]:
        """
        Track CPU utilization, temperature, memory
        
        Returns:
            Dictionary with CPU metrics
        """
        if not psutil:
            return {}
        
        try:
            # Utilization per core
            cpu_util = psutil.cpu_percent(interval=0.1, percpu=True)
            
            # Memory
            vm = psutil.virtual_memory()
            
            # Temperature
            temps = {}
            if self.has_cpu_temp:
                try:
                    temp_sensors = psutil.sensors_temperatures()
                    if 'coretemp' in temp_sensors:
                        # Average core temperature
                        core_temps = [t.current for t in temp_sensors['coretemp']]
                        temps['cpu_avg_temp_c'] = np.mean(core_temps) if core_temps else 0
                        temps['cpu_max_temp_c'] = np.max(core_temps) if core_temps else 0
                except:
                    pass
            
            metrics = {
                'cpu_util_avg_percent': np.mean(cpu_util),
                'cpu_util_max_percent': np.max(cpu_util),
                'cpu_util_min_percent': np.min(cpu_util),
                'ram_used_gb': vm.used / 1e9,
                'ram_total_gb': vm.total / 1e9,
                'ram_percent': vm.percent,
                **temps
            }
            
            # Alert if thresholds exceeded
            if temps.get('cpu_max_temp_c', 0) >= self.CPU_TEMP_WARNING:
                self.logger.warning(f"🟡 WARNING: CPU max temperature {temps['cpu_max_temp_c']:.1f}°C")
            
            return metrics
        
        except Exception as e:
            self.logger.debug(f"Error reading CPU metrics: {e}")
            return {}
    
    def track_training(self, 
                      loss: float,
                      accuracy: Optional[float] = None,
                      batch_time: float = 0.0,
                      learning_rate: float = 0.0,
                      batch_size: int = 64) -> Dict[str, float]:
        """
        Track training metrics
        
        Args:
            loss: Training loss value
            accuracy: Training accuracy (optional)
            batch_time: Time to process batch (seconds)
            learning_rate: Current learning rate
            batch_size: Batch size used
        
        Returns:
            Dictionary with training metrics
        """
        metrics = {
            'loss': loss,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
        }
        
        if accuracy is not None:
            metrics['accuracy'] = accuracy
        
        if batch_time > 0:
            positions_per_sec = batch_size / batch_time
            metrics['batch_time_ms'] = batch_time * 1000
            metrics['positions_per_sec'] = positions_per_sec
        
        return metrics
    
    def log_metrics(self, **kwargs):
        """
        Log metrics to storage
        
        Args:
            **kwargs: Metric name-value pairs
        """
        timestamp = time.time() - self.start_time
        
        for key, value in kwargs.items():
            if value is not None:
                self.metrics[key].append((timestamp, value))
    
    def track_iteration(self,
                       step: int,
                       loss: float,
                       accuracy: Optional[float] = None,
                       batch_time: float = 0.0,
                       learning_rate: float = 0.0,
                       batch_size: int = 64):
        """
        Track a complete training iteration
        
        Args:
            step: Training step number
            loss: Training loss
            accuracy: Training accuracy (optional)
            batch_time: Time to process batch
            learning_rate: Current learning rate
            batch_size: Batch size
        """
        # Collect metrics
        gpu_metrics = self.track_gpu()
        cpu_metrics = self.track_cpu()
        train_metrics = self.track_training(loss, accuracy, batch_time, learning_rate, batch_size)
        
        # Log all metrics
        all_metrics = {**gpu_metrics, **cpu_metrics, **train_metrics}
        self.log_metrics(**all_metrics)
        
        # Log to console every 100 steps
        if step % 100 == 0:
            self._log_summary(step, all_metrics)
    
    def _log_summary(self, step: int, metrics: Dict[str, float]):
        """Log summary of current metrics"""
        msg = f"Step {step:6d} | "
        
        if 'loss' in metrics:
            msg += f"Loss: {metrics['loss']:.4f} | "
        
        if 'positions_per_sec' in metrics:
            msg += f"Speed: {metrics['positions_per_sec']:,.0f} pos/s | "
        
        if 'gpu_mem_percent' in metrics:
            msg += f"GPU Mem: {metrics['gpu_mem_percent']:.0f}% | "
        
        if 'gpu_temp_c' in metrics:
            msg += f"GPU Temp: {metrics['gpu_temp_c']:.0f}°C"
        
        self.logger.info(msg)
    
    def generate_report(self) -> Dict:
        """
        Generate comprehensive performance report
        
        Returns:
            Dictionary with aggregated metrics
        """
        report = {
            'model': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': time.time() - self.start_time,
            'metrics': {}
        }
        
        # Aggregate metrics
        for key, values in self.metrics.items():
            if values:
                data = np.array([v for _, v in values])
                report['metrics'][key] = {
                    'min': float(np.min(data)),
                    'max': float(np.max(data)),
                    'mean': float(np.mean(data)),
                    'median': float(np.median(data)),
                    'std': float(np.std(data)),
                }
        
        return report
    
    def save_report(self) -> Path:
        """Save performance report to JSON file"""
        report = self.generate_report()
        report_path = self.log_dir / f"{self.model_name}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"✅ Report saved: {report_path}")
        return report_path
    
    def save_metrics_csv(self) -> Path:
        """Save metrics to CSV for analysis"""
        import csv
        
        csv_path = self.log_dir / f"{self.model_name}_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Get all unique keys
        all_keys = sorted(set(self.metrics.keys()))
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['timestamp_s'] + all_keys
            writer.writerow(header)
            
            # Find all unique timestamps
            all_timestamps = set()
            for values in self.metrics.values():
                all_timestamps.update([t for t, _ in values])
            
            # Create lookup dictionaries
            lookup = defaultdict(dict)
            for key, values in self.metrics.items():
                for ts, val in values:
                    lookup[ts][key] = val
            
            # Write rows
            for ts in sorted(all_timestamps):
                row = [ts]
                for key in all_keys:
                    row.append(lookup[ts].get(key, ''))
                writer.writerow(row)
        
        self.logger.info(f"✅ Metrics saved to CSV: {csv_path}")
        return csv_path


def create_monitoring_dashboard():
    """
    Create a simple monitoring dashboard showing current status
    
    This can be called periodically to show real-time system status
    """
    monitor = PerformanceMonitor()
    
    print("\n" + "="*80)
    print("🔬 NEURAL NETWORK PERFORMANCE MONITORING")
    print("="*80)
    
    gpu_metrics = monitor.track_gpu()
    cpu_metrics = monitor.track_cpu()
    
    if gpu_metrics:
        print("\n📊 GPU Status:")
        print(f"  Utilization:  {gpu_metrics.get('gpu_util_percent', 0):.1f}%")
        print(f"  Memory:       {gpu_metrics.get('gpu_mem_used_gb', 0):.2f}GB / {gpu_metrics.get('gpu_mem_total_gb', 0):.2f}GB " +
              f"({gpu_metrics.get('gpu_mem_percent', 0):.1f}%)")
        print(f"  Temperature:  {gpu_metrics.get('gpu_temp_c', 0):.1f}°C")
        if gpu_metrics.get('gpu_power_w', 0) > 0:
            print(f"  Power:        {gpu_metrics.get('gpu_power_w', 0):.1f}W / {gpu_metrics.get('gpu_power_limit_w', 0):.1f}W limit")
    
    if cpu_metrics:
        print("\n💻 CPU Status:")
        print(f"  Utilization:  {cpu_metrics.get('cpu_util_avg_percent', 0):.1f}% avg " +
              f"(min: {cpu_metrics.get('cpu_util_min_percent', 0):.1f}%, max: {cpu_metrics.get('cpu_util_max_percent', 0):.1f}%)")
        print(f"  Memory:       {cpu_metrics.get('ram_used_gb', 0):.2f}GB / {cpu_metrics.get('ram_total_gb', 0):.2f}GB " +
              f"({cpu_metrics.get('ram_percent', 0):.1f}%)")
        if 'cpu_avg_temp_c' in cpu_metrics:
            print(f"  Temperature:  {cpu_metrics.get('cpu_avg_temp_c', 0):.1f}°C avg " +
                  f"(max: {cpu_metrics.get('cpu_max_temp_c', 0):.1f}°C)")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # Demo: Show current system status
    create_monitoring_dashboard()
