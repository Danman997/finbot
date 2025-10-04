"""
Кастомные исключения
"""
from typing import Optional

class FinBotException(Exception):
    """Базовое исключение для FinBot"""
    def __init__(self, message: str, error_code: Optional[str] = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class DatabaseError(FinBotException):
    """Ошибка базы данных"""
    pass

class ValidationError(FinBotException):
    """Ошибка валидации данных"""
    pass

class AuthenticationError(FinBotException):
    """Ошибка аутентификации"""
    pass

class AuthorizationError(FinBotException):
    """Ошибка авторизации"""
    pass

class RateLimitError(FinBotException):
    """Ошибка превышения лимита запросов"""
    pass

class ConfigurationError(FinBotException):
    """Ошибка конфигурации"""
    pass
