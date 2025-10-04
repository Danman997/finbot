"""
Сервис для работы с пользователями
"""
from typing import Optional, List
from datetime import datetime
from models.user import User, UserRole
from services.database_service import db_service
from utils import logger, DatabaseError, ValidationError

class UserService:
    """Сервис для работы с пользователями"""
    
    async def create_user(self, user_id: int, username: str, phone: Optional[str] = None, 
                         role: UserRole = UserRole.USER) -> User:
        """Создание нового пользователя"""
        try:
            query = """
                INSERT INTO users (id, username, phone, role, is_active, created_at, last_activity)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING *
            """
            
            now = datetime.now()
            result = await db_service.fetch_one(
                query, user_id, username, phone, role.value, True, now, now
            )
            
            if not result:
                raise DatabaseError("Не удалось создать пользователя")
            
            logger.info(f"Создан пользователь: {username} (ID: {user_id})")
            return User.from_dict(dict(result))
            
        except Exception as e:
            logger.error(f"Ошибка создания пользователя: {e}")
            raise DatabaseError(f"Не удалось создать пользователя: {e}")
    
    async def get_user(self, user_id: int) -> Optional[User]:
        """Получение пользователя по ID"""
        try:
            query = "SELECT * FROM users WHERE id = $1"
            result = await db_service.fetch_one(query, user_id)
            
            if result:
                return User.from_dict(dict(result))
            return None
            
        except Exception as e:
            logger.error(f"Ошибка получения пользователя {user_id}: {e}")
            raise DatabaseError(f"Не удалось получить пользователя: {e}")
    
    async def update_user(self, user_id: int, **kwargs) -> Optional[User]:
        """Обновление пользователя"""
        try:
            # Формируем SET часть запроса
            set_parts = []
            values = []
            param_count = 1
            
            for key, value in kwargs.items():
                if key in ['username', 'phone', 'role', 'is_active', 'folder_name']:
                    set_parts.append(f"{key} = ${param_count}")
                    values.append(value)
                    param_count += 1
            
            if not set_parts:
                raise ValidationError("Нет полей для обновления")
            
            # Добавляем updated_at
            set_parts.append(f"last_activity = ${param_count}")
            values.append(datetime.now())
            param_count += 1
            
            # Добавляем user_id в конец
            values.append(user_id)
            
            query = f"""
                UPDATE users 
                SET {', '.join(set_parts)}
                WHERE id = ${param_count}
                RETURNING *
            """
            
            result = await db_service.fetch_one(query, *values)
            
            if result:
                logger.info(f"Обновлен пользователь: {user_id}")
                return User.from_dict(dict(result))
            return None
            
        except Exception as e:
            logger.error(f"Ошибка обновления пользователя {user_id}: {e}")
            raise DatabaseError(f"Не удалось обновить пользователя: {e}")
    
    async def delete_user(self, user_id: int) -> bool:
        """Удаление пользователя"""
        try:
            query = "DELETE FROM users WHERE id = $1"
            result = await db_service.execute(query, user_id)
            
            if "DELETE 1" in result:
                logger.info(f"Удален пользователь: {user_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Ошибка удаления пользователя {user_id}: {e}")
            raise DatabaseError(f"Не удалось удалить пользователя: {e}")
    
    async def get_all_users(self, limit: int = 100, offset: int = 0) -> List[User]:
        """Получение списка всех пользователей"""
        try:
            query = """
                SELECT * FROM users 
                ORDER BY created_at DESC 
                LIMIT $1 OFFSET $2
            """
            results = await db_service.fetch_all(query, limit, offset)
            
            return [User.from_dict(dict(row)) for row in results]
            
        except Exception as e:
            logger.error(f"Ошибка получения списка пользователей: {e}")
            raise DatabaseError(f"Не удалось получить список пользователей: {e}")
    
    async def get_users_count(self) -> int:
        """Получение количества пользователей"""
        try:
            query = "SELECT COUNT(*) FROM users"
            return await db_service.fetch_val(query)
            
        except Exception as e:
            logger.error(f"Ошибка получения количества пользователей: {e}")
            raise DatabaseError(f"Не удалось получить количество пользователей: {e}")
    
    async def is_user_exists(self, user_id: int) -> bool:
        """Проверка существования пользователя"""
        try:
            query = "SELECT EXISTS(SELECT 1 FROM users WHERE id = $1)"
            return await db_service.fetch_val(query, user_id)
            
        except Exception as e:
            logger.error(f"Ошибка проверки существования пользователя {user_id}: {e}")
            return False

# Глобальный экземпляр сервиса
user_service = UserService()
