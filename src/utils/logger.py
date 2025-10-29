import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime

def setup_logger(name: str = __name__, 
                level: int = logging.INFO,
                log_file: Optional[str] = None,
                format_string: Optional[str] = None,
                propagate: bool = False) -> logging.Logger:
    """
    Set up a logger with consistent formatting and handlers.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file to log to
        format_string: Custom format string
        propagate: Whether to propagate to parent loggers
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    logger.propagate = propagate
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'logger': record.name,
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)

def setup_structured_logger(name: str = __name__,
                          level: int = logging.INFO,
                          log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with JSON formatting for structured logging.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file to log to
    
    Returns:
        Configured logger with JSON formatting
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    logger.propagate = False
    
    formatter = JSONFormatter()
    
    # Console handler with JSON
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

class ProgressLogger:
    """Utility class for logging progress with timing information."""
    
    def __init__(self, logger: logging.Logger, total_steps: int, description: str = ""):
        self.logger = logger
        self.total_steps = total_steps
        self.description = description
        self.current_step = 0
        self.start_time = datetime.now()
    
    def step(self, message: str = "") -> None:
        """Log progress for current step."""
        self.current_step += 1
        elapsed = (datetime.now() - self.start_time).total_seconds()
        progress = (self.current_step / self.total_steps) * 100
        
        log_message = f"{self.description}: {progress:.1f}% ({self.current_step}/{self.total_steps})"
        if message:
            log_message += f" - {message}"
        
        if self.current_step == 1:
            # First step - log start
            self.logger.info(f"Starting {self.description}")
            self.logger.info(f"Estimated total steps: {self.total_steps}")
        elif self.current_step == self.total_steps:
            # Last step - log completion
            self.logger.info(f"Completed {self.description} in {elapsed:.2f}s")
        else:
            # Intermediate step
            self.logger.debug(log_message)
    
    def get_eta(self) -> str:
        """Get estimated time of arrival/completion."""
        if self.current_step == 0:
            return "Unknown"
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        time_per_step = elapsed / self.current_step
        remaining_time = time_per_step * (self.total_steps - self.current_step)
        
        if remaining_time < 60:
            return f"{remaining_time:.0f}s"
        elif remaining_time < 3600:
            return f"{remaining_time/60:.1f}m"
        else:
            return f"{remaining_time/3600:.1f}h"

def log_execution_time(logger: logging.Logger):
    """Decorator to log function execution time."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            logger.debug(f"Starting {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.debug(f"Completed {func.__name__} in {execution_time:.2f}s")
                return result
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.error(f"Failed {func.__name__} after {execution_time:.2f}s: {e}")
                raise
        
        return wrapper
    return decorator

def setup_experiment_logging(experiment_name: str, 
                           log_dir: str = "logs") -> Dict[str, logging.Logger]:
    """
    Set up comprehensive logging for an experiment.
    
    Args:
        experiment_name: Name of the experiment
        log_dir: Directory for log files
    
    Returns:
        Dictionary of configured loggers
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/{experiment_name}_{timestamp}.log"
    
    loggers = {
        'experiment': setup_logger(f"experiment.{experiment_name}", log_file=log_file),
        'data': setup_logger(f"data.{experiment_name}", log_file=log_file),
        'model': setup_logger(f"model.{experiment_name}", log_file=log_file),
        'training': setup_logger(f"training.{experiment_name}", log_file=log_file),
        'evaluation': setup_logger(f"evaluation.{experiment_name}", log_file=log_file)
    }
    
    # Log experiment start
    loggers['experiment'].info(f"Starting experiment: {experiment_name}")
    loggers['experiment'].info(f"Log file: {log_file}")
    
    return loggers