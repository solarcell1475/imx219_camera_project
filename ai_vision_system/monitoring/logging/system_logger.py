#!/usr/bin/env python3
"""
System Logging
==============
Structured logging with JSON format, log rotation, and error tracking.
"""

import logging
import json
import os
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Dict, Optional


class SystemLogger:
    """System logging with JSON support"""
    
    def __init__(self, log_dir: str = "logs", log_level: int = logging.INFO):
        """
        Initialize logger
        
        Args:
            log_dir: Directory for log files
            log_level: Logging level
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create loggers
        self.logger = logging.getLogger('yolo_vision_system')
        self.logger.setLevel(log_level)
        
        # Performance logger
        self.perf_logger = logging.getLogger('yolo_performance')
        self.perf_logger.setLevel(logging.INFO)
        
        # Error logger
        self.error_logger = logging.getLogger('yolo_errors')
        self.error_logger.setLevel(logging.ERROR)
        
        # Setup handlers
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup logging handlers"""
        # Main log file
        main_handler = RotatingFileHandler(
            self.log_dir / "system.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        main_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(main_handler)
        
        # Performance log file (JSON)
        perf_handler = RotatingFileHandler(
            self.log_dir / "performance.log",
            maxBytes=10*1024*1024,
            backupCount=5
        )
        perf_handler.setFormatter(
            logging.Formatter('%(message)s')  # JSON messages
        )
        self.perf_logger.addHandler(perf_handler)
        
        # Error log file
        error_handler = RotatingFileHandler(
            self.log_dir / "errors.log",
            maxBytes=10*1024*1024,
            backupCount=10
        )
        error_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s\n%(exc_info)s')
        )
        self.error_logger.addHandler(error_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(console_handler)
    
    def log_info(self, message: str, **kwargs):
        """Log info message"""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.info(message)
    
    def log_warning(self, message: str, **kwargs):
        """Log warning message"""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.warning(message)
    
    def log_error(self, message: str, exc_info=None, **kwargs):
        """Log error message"""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.error_logger.error(message, exc_info=exc_info)
    
    def log_performance(self, metrics: Dict):
        """Log performance metrics as JSON"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        self.perf_logger.info(json.dumps(log_entry))
    
    def log_inference_result(self, detections: list, inference_time: float, **kwargs):
        """Log inference result"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'detection_count': len(detections),
            'inference_time_ms': inference_time,
            **kwargs
        }
        self.perf_logger.info(json.dumps(log_entry))
    
    def log_system_event(self, event_type: str, message: str, **kwargs):
        """Log system event"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'message': message,
            **kwargs
        }
        self.logger.info(f"EVENT: {json.dumps(log_entry)}")
