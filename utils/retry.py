"""
Утилиты для повторных попыток выполнения операций
"""
import functools
import time
import logging
from typing import Callable, Type, Tuple, Optional
import asyncio

logger = logging.getLogger(__name__)

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable] = None
):
    """
    Декоратор для повторных попыток выполнения функции при ошибках
    
    Args:
        max_attempts: Максимальное количество попыток
        delay: Начальная задержка между попытками (сек)
        backoff: Множитель увеличения задержки
        exceptions: Кортеж исключений, при которых делать retry
        on_retry: Callback функция, вызываемая при повторной попытке
    
    Example:
        @retry(max_attempts=3, delay=1.0, exceptions=(ConnectionError,))
        async def fetch_data():
            return await api_call()
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts:
                        logger.error(
                            f"Function {func.__name__} failed after {max_attempts} attempts. "
                            f"Last error: {e}"
                        )
                        raise
                    
                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt}/{max_attempts}). "
                        f"Retrying in {current_delay}s. Error: {e}"
                    )
                    
                    if on_retry:
                        on_retry(attempt, e)
                    
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            
            raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts:
                        logger.error(
                            f"Function {func.__name__} failed after {max_attempts} attempts. "
                            f"Last error: {e}"
                        )
                        raise
                    
                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt}/{max_attempts}). "
                        f"Retrying in {current_delay}s. Error: {e}"
                    )
                    
                    if on_retry:
                        on_retry(attempt, e)
                    
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            raise last_exception
        
        # Определяем, является ли функция асинхронной
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator

class CircuitBreaker:
    """
    Circuit Breaker для предотвращения каскадных сбоев
    
    Состояния:
    - CLOSED: Нормальная работа
    - OPEN: Блокировка вызовов после превышения лимита ошибок
    - HALF_OPEN: Пробное восстановление
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs):
        """Вызывает функцию через circuit breaker"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.info(f"Circuit breaker for {func.__name__} entering HALF_OPEN state")
            else:
                raise Exception(f"Circuit breaker is OPEN for {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Обрабатывает успешный вызов"""
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            logger.info("Circuit breaker entering CLOSED state")
        self.failure_count = 0
    
    def _on_failure(self):
        """Обрабатывает неудачный вызов"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.error(
                f"Circuit breaker entering OPEN state after {self.failure_count} failures"
            )

def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: Type[Exception] = Exception
):
    """
    Декоратор circuit breaker
    
    Example:
        @circuit_breaker(failure_threshold=5, recovery_timeout=60)
        def external_api_call():
            return requests.get("https://api.example.com")
    """
    breaker = CircuitBreaker(failure_threshold, recovery_timeout, expected_exception)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        return wrapper
    
    return decorator

