"""
Сервис для работы с базой данных
"""
import asyncio
import asyncpg
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from config import config
from utils import logger, DatabaseError

class DatabaseService:
    """Сервис для работы с базой данных"""
    
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
    
    async def initialize(self):
        """Инициализация пула соединений"""
        try:
            self.pool = await asyncpg.create_pool(
                host=config.database.host,
                port=config.database.port,
                database=config.database.name,
                user=config.database.user,
                password=config.database.password,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            logger.info("✅ Пул соединений с базой данных создан")
        except Exception as e:
            logger.error(f"❌ Ошибка создания пула соединений: {e}")
            raise DatabaseError(f"Не удалось подключиться к базе данных: {e}")
    
    async def close(self):
        """Закрытие пула соединений"""
        if self.pool:
            await self.pool.close()
            logger.info("Пул соединений закрыт")
    
    @asynccontextmanager
    async def get_connection(self):
        """Получение соединения из пула"""
        if not self.pool:
            raise DatabaseError("Пул соединений не инициализирован")
        
        async with self.pool.acquire() as connection:
            yield connection
    
    async def execute(self, query: str, *args) -> str:
        """Выполнение запроса без возврата результата"""
        async with self.get_connection() as conn:
            return await conn.execute(query, *args)
    
    async def fetch_one(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """Получение одной записи"""
        async with self.get_connection() as conn:
            return await conn.fetchrow(query, *args)
    
    async def fetch_all(self, query: str, *args) -> List[Dict[str, Any]]:
        """Получение всех записей"""
        async with self.get_connection() as conn:
            return await conn.fetch(query, *args)
    
    async def fetch_val(self, query: str, *args) -> Any:
        """Получение одного значения"""
        async with self.get_connection() as conn:
            return await conn.fetchval(query, *args)
    
    async def transaction(self):
        """Контекстный менеджер для транзакций"""
        return self.get_connection()

# Глобальный экземпляр сервиса
db_service = DatabaseService()
