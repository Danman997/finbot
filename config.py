"""
Конфигурация приложения
"""
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

@dataclass
class DatabaseConfig:
    """Конфигурация базы данных"""
    host: str
    port: int
    name: str
    user: str
    password: str
    
    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

@dataclass
class BotConfig:
    """Конфигурация бота"""
    token: str
    admin_ids: list[int]
    max_users_per_group: int = 5
    rate_limit_per_minute: int = 30

@dataclass
class AppConfig:
    """Основная конфигурация приложения"""
    database: DatabaseConfig
    bot: BotConfig
    debug: bool = False
    log_level: str = "INFO"

def load_config() -> AppConfig:
    """Загрузка конфигурации из переменных окружения"""
    
    # Проверяем обязательные переменные
    required_vars = {
        'BOT_TOKEN': os.getenv('BOT_TOKEN'),
        'DATABASE_HOST': os.getenv('DATABASE_HOST'),
        'DATABASE_NAME': os.getenv('DATABASE_NAME'),
        'DATABASE_USER': os.getenv('DATABASE_USER'),
        'DATABASE_PASSWORD': os.getenv('DATABASE_PASSWORD'),
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        raise ValueError(f"Отсутствуют обязательные переменные окружения: {missing_vars}")
    
    # Создаем конфигурацию базы данных
    database_config = DatabaseConfig(
        host=required_vars['DATABASE_HOST'],
        port=int(os.getenv('DATABASE_PORT', '5432')),
        name=required_vars['DATABASE_NAME'],
        user=required_vars['DATABASE_USER'],
        password=required_vars['DATABASE_PASSWORD']
    )
    
    # Создаем конфигурацию бота
    bot_config = BotConfig(
        token=required_vars['BOT_TOKEN'],
        admin_ids=[int(x) for x in os.getenv('ADMIN_IDS', '').split(',') if x.strip()]
    )
    
    return AppConfig(
        database=database_config,
        bot=bot_config,
        debug=os.getenv('DEBUG', 'False').lower() == 'true',
        log_level=os.getenv('LOG_LEVEL', 'INFO')
    )

# Глобальная конфигурация
config = load_config()
