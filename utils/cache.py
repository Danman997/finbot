"""
Система кэширования для повышения производительности
"""
import functools
import time
from typing import Any, Callable, Dict, Tuple, Optional
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

class SimpleCache:
    """Простой LRU кэш с поддержкой TTL"""
    
    def __init__(self, ttl: int = 300, max_size: int = 1000):
        """
        Args:
            ttl: Time To Live в секундах
            max_size: Максимальный размер кэша
        """
        self.ttl = ttl
        self.max_size = max_size
        self.cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Получает значение из кэша"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                # Перемещаем в конец (LRU)
                self.cache.move_to_end(key)
                self.hits += 1
                logger.debug(f"Cache hit: {key}")
                return value
            else:
                # Удаляем устаревшую запись
                del self.cache[key]
                logger.debug(f"Cache expired: {key}")
        
        self.misses += 1
        logger.debug(f"Cache miss: {key}")
        return None
    
    def set(self, key: str, value: Any):
        """Сохраняет значение в кэш"""
        if key in self.cache:
            # Обновляем существующее значение
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.max_size:
            # Удаляем самое старое значение (LRU)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            logger.debug(f"Cache evicted (LRU): {oldest_key}")
        
        self.cache[key] = (value, time.time())
        logger.debug(f"Cache set: {key}")
    
    def delete(self, key: str):
        """Удаляет значение из кэша"""
        if key in self.cache:
            del self.cache[key]
            logger.debug(f"Cache deleted: {key}")
    
    def clear(self):
        """Очищает весь кэш"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, int]:
        """Возвращает статистику кэша"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': round(hit_rate, 2),
            'size': len(self.cache),
            'max_size': self.max_size
        }

# Глобальный кэш
_global_cache = SimpleCache()

def cached(ttl: int = 300, key_prefix: str = ""):
    """
    Декоратор для кэширования результатов функций
    
    Args:
        ttl: Time To Live в секундах
        key_prefix: Префикс для ключа кэша
    
    Example:
        @cached(ttl=600, key_prefix="user")
        def get_user_data(user_id: int):
            return expensive_database_query(user_id)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Создаем ключ кэша
            cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Проверяем кэш
            cached_result = _global_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Выполняем функцию и кэшируем результат
            result = func(*args, **kwargs)
            _global_cache.set(cache_key, result)
            return result
        
        # Добавляем методы для управления кэшем
        wrapper.cache_clear = lambda: _global_cache.clear()
        wrapper.cache_stats = lambda: _global_cache.get_stats()
        
        return wrapper
    return decorator

def cache_key(func_name: str, *args, **kwargs) -> str:
    """Генерирует ключ кэша из имени функции и аргументов"""
    return f"{func_name}:{hash(str(args) + str(kwargs))}"

def get_cache_stats() -> Dict[str, int]:
    """Возвращает глобальную статистику кэша"""
    return _global_cache.get_stats()

def clear_cache():
    """Очищает глобальный кэш"""
    _global_cache.clear()

