"""
Сервис для работы с напоминаниями
"""
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from decimal import Decimal
from models.reminder import Reminder
from services.database_service import db_service
from utils import logger, DatabaseError, ValidationError

class ReminderService:
    """Сервис для работы с напоминаниями"""
    
    async def create_reminder(self, user_id: int, title: str, description: Optional[str], 
                             amount: Decimal, start_date: date, end_date: Optional[date] = None) -> Reminder:
        """Создание нового напоминания"""
        try:
            query = """
                INSERT INTO reminders (user_id, title, description, amount, start_date, end_date, is_active, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING *
            """
            
            now = datetime.now()
            result = await db_service.fetch_one(
                query, user_id, title, description, amount, start_date, end_date, True, now, now
            )
            
            if not result:
                raise DatabaseError("Не удалось создать напоминание")
            
            logger.info(f"Создано напоминание: {title} для пользователя {user_id}")
            return Reminder.from_dict(dict(result))
            
        except Exception as e:
            logger.error(f"Ошибка создания напоминания: {e}")
            raise DatabaseError(f"Не удалось создать напоминание: {e}")
    
    async def get_reminder(self, reminder_id: int) -> Optional[Reminder]:
        """Получение напоминания по ID"""
        try:
            query = "SELECT * FROM reminders WHERE id = $1"
            result = await db_service.fetch_one(query, reminder_id)
            
            if result:
                return Reminder.from_dict(dict(result))
            return None
            
        except Exception as e:
            logger.error(f"Ошибка получения напоминания {reminder_id}: {e}")
            raise DatabaseError(f"Не удалось получить напоминание: {e}")
    
    async def update_reminder(self, reminder_id: int, **kwargs) -> Optional[Reminder]:
        """Обновление напоминания"""
        try:
            # Формируем SET часть запроса
            set_parts = []
            values = []
            param_count = 1
            
            for key, value in kwargs.items():
                if key in ['title', 'description', 'amount', 'start_date', 'end_date', 'is_active']:
                    set_parts.append(f"{key} = ${param_count}")
                    values.append(value)
                    param_count += 1
            
            if not set_parts:
                raise ValidationError("Нет полей для обновления")
            
            # Добавляем updated_at
            set_parts.append(f"updated_at = ${param_count}")
            values.append(datetime.now())
            param_count += 1
            
            # Добавляем reminder_id в конец
            values.append(reminder_id)
            
            query = f"""
                UPDATE reminders 
                SET {', '.join(set_parts)}
                WHERE id = ${param_count}
                RETURNING *
            """
            
            result = await db_service.fetch_one(query, *values)
            
            if result:
                logger.info(f"Обновлено напоминание: {reminder_id}")
                return Reminder.from_dict(dict(result))
            return None
            
        except Exception as e:
            logger.error(f"Ошибка обновления напоминания {reminder_id}: {e}")
            raise DatabaseError(f"Не удалось обновить напоминание: {e}")
    
    async def delete_reminder(self, reminder_id: int) -> bool:
        """Удаление напоминания"""
        try:
            query = "DELETE FROM reminders WHERE id = $1"
            result = await db_service.execute(query, reminder_id)
            
            if "DELETE 1" in result:
                logger.info(f"Удалено напоминание: {reminder_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Ошибка удаления напоминания {reminder_id}: {e}")
            raise DatabaseError(f"Не удалось удалить напоминание: {e}")
    
    async def get_user_reminders(self, user_id: int, active_only: bool = True, 
                                limit: int = 100, offset: int = 0) -> List[Reminder]:
        """Получение напоминаний пользователя"""
        try:
            query = """
                SELECT * FROM reminders 
                WHERE user_id = $1
            """
            params = [user_id]
            param_count = 1
            
            if active_only:
                param_count += 1
                query += f" AND is_active = ${param_count}"
                params.append(True)
            
            query += f" ORDER BY start_date ASC, created_at DESC LIMIT ${param_count + 1} OFFSET ${param_count + 2}"
            params.extend([limit, offset])
            
            results = await db_service.fetch_all(query, *params)
            return [Reminder.from_dict(dict(row)) for row in results]
            
        except Exception as e:
            logger.error(f"Ошибка получения напоминаний пользователя {user_id}: {e}")
            raise DatabaseError(f"Не удалось получить напоминания: {e}")
    
    async def get_active_reminders_for_date(self, target_date: date) -> List[Reminder]:
        """Получение активных напоминаний на определенную дату"""
        try:
            query = """
                SELECT * FROM reminders 
                WHERE is_active = $1 
                AND start_date <= $2 
                AND (end_date IS NULL OR end_date >= $2)
                ORDER BY start_date ASC
            """
            
            results = await db_service.fetch_all(query, True, target_date)
            return [Reminder.from_dict(dict(row)) for row in results]
            
        except Exception as e:
            logger.error(f"Ошибка получения напоминаний на дату {target_date}: {e}")
            raise DatabaseError(f"Не удалось получить напоминания: {e}")
    
    async def get_reminders_summary(self, user_id: int) -> Dict[str, Any]:
        """Получение сводки по напоминаниям"""
        try:
            query = """
                SELECT 
                    COUNT(*) as total_count,
                    COUNT(CASE WHEN is_active THEN 1 END) as active_count,
                    SUM(CASE WHEN is_active THEN amount ELSE 0 END) as total_amount
                FROM reminders 
                WHERE user_id = $1
            """
            
            result = await db_service.fetch_one(query, user_id)
            
            if result:
                return {
                    'total_count': result['total_count'],
                    'active_count': result['active_count'],
                    'total_amount': Decimal(str(result['total_amount'] or 0))
                }
            
            return {
                'total_count': 0,
                'active_count': 0,
                'total_amount': Decimal('0')
            }
            
        except Exception as e:
            logger.error(f"Ошибка получения сводки напоминаний: {e}")
            raise DatabaseError(f"Не удалось получить сводку напоминаний: {e}")
    
    async def deactivate_expired_reminders(self) -> int:
        """Деактивация просроченных напоминаний"""
        try:
            query = """
                UPDATE reminders 
                SET is_active = $1, updated_at = $2
                WHERE is_active = $3 
                AND end_date IS NOT NULL 
                AND end_date < $4
                RETURNING id
            """
            
            today = date.today()
            results = await db_service.fetch_all(query, False, datetime.now(), True, today)
            
            count = len(results)
            if count > 0:
                logger.info(f"Деактивировано {count} просроченных напоминаний")
            
            return count
            
        except Exception as e:
            logger.error(f"Ошибка деактивации просроченных напоминаний: {e}")
            raise DatabaseError(f"Не удалось деактивировать просроченные напоминания: {e}")

# Глобальный экземпляр сервиса
reminder_service = ReminderService()
