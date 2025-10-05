"""
Улучшенная конфигурация приложения с валидацией и типизацией
"""
import os
from dataclasses import dataclass, field
from typing import Optional, List
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Загружаем переменные окружения
load_dotenv()

@dataclass
class DatabaseConfig:
    """Конфигурация базы данных"""
    host: Optional[str] = None
    port: str = "5432"
    name: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    
    @property
    def url(self) -> Optional[str]:
        """Генерирует URL для подключения к базе данных"""
        if all([self.host, self.name, self.user, self.password]):
            return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
        return None
    
    @property
    def is_configured(self) -> bool:
        """Проверяет, настроена ли база данных"""
        return self.url is not None

@dataclass
class BotConfig:
    """Конфигурация бота"""
    token: str
    admin_id: int = 0
    max_users_per_group: int = 5
    max_expense_amount: float = 10000000.0  # 10 миллионов
    rate_limit_requests: int = 30
    rate_limit_window: int = 60  # секунд
    
    def validate(self) -> List[str]:
        """Валидация конфигурации бота"""
        errors = []
        if not self.token:
            errors.append("BOT_TOKEN is required")
        if self.admin_id <= 0:
            errors.append("ADMIN_ID must be a positive integer")
        if self.max_users_per_group <= 0:
            errors.append("max_users_per_group must be positive")
        return errors

@dataclass
class CacheConfig:
    """Конфигурация кэширования"""
    enabled: bool = True
    ttl: int = 300  # 5 минут
    max_size: int = 1000

@dataclass
class LoggingConfig:
    """Конфигурация логирования"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "logs/finbot.log"
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5

class Settings:
    """Основные настройки приложения"""
    
    def __init__(self):
        self.bot = BotConfig(
            token=os.environ.get('BOT_TOKEN', ''),
            admin_id=int(os.environ.get('ADMIN_ID', os.environ.get('admin', '0')))
        )
        
        self.database = DatabaseConfig(
            host=os.environ.get('DATABASE_HOST'),
            port=os.environ.get('DATABASE_PORT', '5432'),
            name=os.environ.get('DATABASE_NAME'),
            user=os.environ.get('DATABASE_USER'),
            password=os.environ.get('DATABASE_PASSWORD')
        )
        
        self.cache = CacheConfig(
            enabled=os.environ.get('CACHE_ENABLED', 'true').lower() == 'true',
            ttl=int(os.environ.get('CACHE_TTL', '300')),
            max_size=int(os.environ.get('CACHE_MAX_SIZE', '1000'))
        )
        
        self.logging = LoggingConfig(
            level=os.environ.get('LOG_LEVEL', 'INFO'),
            file_path=os.environ.get('LOG_FILE', 'logs/finbot.log')
        )
        
        self.debug = os.environ.get('DEBUG', 'False').lower() == 'true'
        
        # Валидация конфигурации
        self._validate()
    
    def _validate(self):
        """Валидация всех настроек"""
        bot_errors = self.bot.validate()
        if bot_errors:
            for error in bot_errors:
                logger.error(f"Configuration error: {error}")
            if not self.debug:
                raise ValueError(f"Configuration validation failed: {bot_errors}")
        
        if not self.database.is_configured:
            logger.warning("⚠️ Database is not configured. Some features may not work.")
    
    def get_database_url(self) -> Optional[str]:
        """Получает URL базы данных"""
        return self.database.url
    
    def is_admin(self, user_id: int) -> bool:
        """Проверяет, является ли пользователь администратором"""
        return user_id == self.bot.admin_id

# Глобальный экземпляр настроек
settings = Settings()

# Логируем статус конфигурации
logger.info(f"✅ Configuration loaded. Debug mode: {settings.debug}")
if settings.database.is_configured:
    logger.info("✅ Database configuration loaded")
else:
    logger.warning("⚠️ Database not configured")

