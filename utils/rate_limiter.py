"""
Rate limiting для защиты от спама
"""
import time
from typing import Dict, Optional
from collections import defaultdict, deque
from utils.exceptions import RateLimitError
from utils.logger import logger

class RateLimiter:
    """Простой rate limiter на основе sliding window"""
    
    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[int, deque] = defaultdict(deque)
    
    def is_allowed(self, user_id: int) -> bool:
        """Проверяет, разрешен ли запрос для пользователя"""
        now = time.time()
        user_requests = self.requests[user_id]
        
        # Удаляем старые запросы
        while user_requests and user_requests[0] <= now - self.window_seconds:
            user_requests.popleft()
        
        # Проверяем лимит
        if len(user_requests) >= self.max_requests:
            logger.warning(f"Rate limit exceeded for user {user_id}")
            return False
        
        # Добавляем текущий запрос
        user_requests.append(now)
        return True
    
    def get_remaining_requests(self, user_id: int) -> int:
        """Возвращает количество оставшихся запросов"""
        now = time.time()
        user_requests = self.requests[user_id]
        
        # Удаляем старые запросы
        while user_requests and user_requests[0] <= now - self.window_seconds:
            user_requests.popleft()
        
        return max(0, self.max_requests - len(user_requests))
    
    def reset_user(self, user_id: int):
        """Сбрасывает лимит для пользователя"""
        if user_id in self.requests:
            del self.requests[user_id]

# Глобальный rate limiter
rate_limiter = RateLimiter()

def check_rate_limit(user_id: int) -> None:
    """Проверяет rate limit и выбрасывает исключение при превышении"""
    if not rate_limiter.is_allowed(user_id):
        remaining = rate_limiter.get_remaining_requests(user_id)
        raise RateLimitError(f"Превышен лимит запросов. Попробуйте через минуту. Осталось: {remaining}")
