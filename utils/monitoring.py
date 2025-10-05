"""
Мониторинг и метрики для отслеживания производительности
"""
import time
import functools
import logging
from typing import Callable, Dict, Any
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Монитор производительности функций"""
    
    def __init__(self):
        self.metrics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'calls': 0,
            'total_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'errors': 0,
            'last_called': None
        })
    
    def record_call(self, func_name: str, execution_time: float, error: bool = False):
        """Записывает метрики вызова функции"""
        metrics = self.metrics[func_name]
        metrics['calls'] += 1
        metrics['total_time'] += execution_time
        metrics['min_time'] = min(metrics['min_time'], execution_time)
        metrics['max_time'] = max(metrics['max_time'], execution_time)
        if error:
            metrics['errors'] += 1
        metrics['last_called'] = datetime.now()
    
    def get_metrics(self, func_name: str = None) -> Dict[str, Any]:
        """Получает метрики функции или всех функций"""
        if func_name:
            metrics = self.metrics.get(func_name, {})
            if metrics and metrics['calls'] > 0:
                metrics['avg_time'] = metrics['total_time'] / metrics['calls']
            return metrics
        
        # Возвращаем метрики всех функций
        result = {}
        for name, metrics in self.metrics.items():
            if metrics['calls'] > 0:
                metrics_copy = metrics.copy()
                metrics_copy['avg_time'] = metrics['total_time'] / metrics['calls']
                result[name] = metrics_copy
        return result
    
    def reset(self, func_name: str = None):
        """Сбрасывает метрики"""
        if func_name:
            if func_name in self.metrics:
                del self.metrics[func_name]
        else:
            self.metrics.clear()
    
    def get_summary(self) -> str:
        """Возвращает краткую сводку по метрикам"""
        lines = ["📊 Performance Metrics Summary:"]
        for func_name, metrics in sorted(self.metrics.items()):
            if metrics['calls'] > 0:
                avg_time = metrics['total_time'] / metrics['calls']
                error_rate = (metrics['errors'] / metrics['calls'] * 100) if metrics['calls'] > 0 else 0
                lines.append(
                    f"  {func_name}:"
                    f" calls={metrics['calls']}"
                    f" avg={avg_time:.3f}s"
                    f" min={metrics['min_time']:.3f}s"
                    f" max={metrics['max_time']:.3f}s"
                    f" errors={metrics['errors']} ({error_rate:.1f}%)"
                )
        return "\n".join(lines)

# Глобальный монитор
_monitor = PerformanceMonitor()

def monitor_performance(func: Callable = None, log_calls: bool = True):
    """
    Декоратор для мониторинга производительности функций
    
    Args:
        func: Функция для мониторинга
        log_calls: Логировать ли каждый вызов
    
    Example:
        @monitor_performance
        async def process_expense(user_id: int, amount: float):
            # ... expensive operation ...
            pass
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            error_occurred = False
            
            try:
                result = await f(*args, **kwargs)
                return result
            except Exception as e:
                error_occurred = True
                logger.error(f"Error in {f.__name__}: {e}")
                raise
            finally:
                execution_time = time.time() - start_time
                _monitor.record_call(f.__name__, execution_time, error_occurred)
                
                if log_calls:
                    status = "ERROR" if error_occurred else "OK"
                    logger.info(
                        f"{status} | {f.__name__} | "
                        f"time={execution_time:.3f}s | "
                        f"args={len(args)} kwargs={len(kwargs)}"
                    )
        
        @functools.wraps(f)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            error_occurred = False
            
            try:
                result = f(*args, **kwargs)
                return result
            except Exception as e:
                error_occurred = True
                logger.error(f"Error in {f.__name__}: {e}")
                raise
            finally:
                execution_time = time.time() - start_time
                _monitor.record_call(f.__name__, execution_time, error_occurred)
                
                if log_calls:
                    status = "ERROR" if error_occurred else "OK"
                    logger.info(
                        f"{status} | {f.__name__} | "
                        f"time={execution_time:.3f}s | "
                        f"args={len(args)} kwargs={len(kwargs)}"
                    )
        
        # Определяем, является ли функция асинхронной
        if hasattr(f, '__code__') and f.__code__.co_flags & 0x80:
            return async_wrapper
        return sync_wrapper
    
    # Позволяет использовать как @monitor_performance и @monitor_performance()
    if func is None:
        return decorator
    return decorator(func)

def get_metrics(func_name: str = None) -> Dict[str, Any]:
    """Получает метрики производительности"""
    return _monitor.get_metrics(func_name)

def get_summary() -> str:
    """Получает сводку по метрикам"""
    return _monitor.get_summary()

def reset_metrics(func_name: str = None):
    """Сбрасывает метрики"""
    _monitor.reset(func_name)

