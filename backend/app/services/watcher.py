# backend/app/services/watcher.py
import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from typing import Callable
import logging

logger = logging.getLogger(__name__)

class ModelFileHandler(FileSystemEventHandler):
    def __init__(self, model_service, model_path):
        self.model_service = model_service
        self.model_path = model_path
        self.last_triggered = 0
        self.debounce_seconds = 5  # Prevent rapid reloads

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(('.bin', '.h5', '.pt')):
            current_time = time.time()
            if current_time - self.last_triggered > self.debounce_seconds:
                self.last_triggered = current_time
                logger.info(f"Detected model file change: {event.src_path}")
                try:
                    self.model_service.load_model()
                    logger.info("Model reloaded successfully")
                except Exception as e:
                    logger.error(f"Failed to reload model: {str(e)}")

def start_watcher(model_service: Callable, model_path: str):
    """Start watching the model directory for changes"""
    if not os.path.exists(model_path):
        logger.warning(f"Model path {model_path} does not exist, not starting watcher")
        return

    event_handler = ModelFileHandler(model_service, model_path)
    observer = Observer()
    observer.schedule(event_handler, path=model_path, recursive=True)
    observer.start()
    logger.info(f"Started model watcher on {model_path}")

    return observer