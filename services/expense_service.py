"""
Сервис для работы с расходами
"""
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from decimal import Decimal
from models.expense import Expense
from services.database_service import db_service
from utils import logger, DatabaseError, ValidationError

class ExpenseService:
    """Сервис для работы с расходами"""
    
    async def create_expense(self, user_id: int, amount: Decimal, description: str, 
                           category: str, transaction_date: Optional[date] = None) -> Expense:
        """Создание нового расхода"""
        try:
            if not transaction_date:
                transaction_date = date.today()
            
            query = """
                INSERT INTO expenses (user_id, amount, description, category, transaction_date, created_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING *
            """
            
            now = datetime.now()
            result = await db_service.fetch_one(
                query, user_id, amount, description, category, transaction_date, now
            )
            
            if not result:
                raise DatabaseError("Не удалось создать расход")
            
            logger.info(f"Создан расход: {description} ({amount} Тг) для пользователя {user_id}")
            return Expense.from_dict(dict(result))
            
        except Exception as e:
            logger.error(f"Ошибка создания расхода: {e}")
            raise DatabaseError(f"Не удалось создать расход: {e}")
    
    async def get_expense(self, expense_id: int) -> Optional[Expense]:
        """Получение расхода по ID"""
        try:
            query = "SELECT * FROM expenses WHERE id = $1"
            result = await db_service.fetch_one(query, expense_id)
            
            if result:
                return Expense.from_dict(dict(result))
            return None
            
        except Exception as e:
            logger.error(f"Ошибка получения расхода {expense_id}: {e}")
            raise DatabaseError(f"Не удалось получить расход: {e}")
    
    async def update_expense(self, expense_id: int, **kwargs) -> Optional[Expense]:
        """Обновление расхода"""
        try:
            # Формируем SET часть запроса
            set_parts = []
            values = []
            param_count = 1
            
            for key, value in kwargs.items():
                if key in ['amount', 'description', 'category', 'transaction_date']:
                    set_parts.append(f"{key} = ${param_count}")
                    values.append(value)
                    param_count += 1
            
            if not set_parts:
                raise ValidationError("Нет полей для обновления")
            
            # Добавляем expense_id в конец
            values.append(expense_id)
            
            query = f"""
                UPDATE expenses 
                SET {', '.join(set_parts)}
                WHERE id = ${param_count}
                RETURNING *
            """
            
            result = await db_service.fetch_one(query, *values)
            
            if result:
                logger.info(f"Обновлен расход: {expense_id}")
                return Expense.from_dict(dict(result))
            return None
            
        except Exception as e:
            logger.error(f"Ошибка обновления расхода {expense_id}: {e}")
            raise DatabaseError(f"Не удалось обновить расход: {e}")
    
    async def delete_expense(self, expense_id: int) -> bool:
        """Удаление расхода"""
        try:
            query = "DELETE FROM expenses WHERE id = $1"
            result = await db_service.execute(query, expense_id)
            
            if "DELETE 1" in result:
                logger.info(f"Удален расход: {expense_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Ошибка удаления расхода {expense_id}: {e}")
            raise DatabaseError(f"Не удалось удалить расход: {e}")
    
    async def get_user_expenses(self, user_id: int, start_date: Optional[date] = None, 
                               end_date: Optional[date] = None, limit: int = 100, 
                               offset: int = 0) -> List[Expense]:
        """Получение расходов пользователя"""
        try:
            query = """
                SELECT * FROM expenses 
                WHERE user_id = $1
            """
            params = [user_id]
            param_count = 1
            
            if start_date:
                param_count += 1
                query += f" AND transaction_date >= ${param_count}"
                params.append(start_date)
            
            if end_date:
                param_count += 1
                query += f" AND transaction_date <= ${param_count}"
                params.append(end_date)
            
            query += f" ORDER BY transaction_date DESC, created_at DESC LIMIT ${param_count + 1} OFFSET ${param_count + 2}"
            params.extend([limit, offset])
            
            results = await db_service.fetch_all(query, *params)
            return [Expense.from_dict(dict(row)) for row in results]
            
        except Exception as e:
            logger.error(f"Ошибка получения расходов пользователя {user_id}: {e}")
            raise DatabaseError(f"Не удалось получить расходы: {e}")
    
    async def get_expenses_by_category(self, user_id: int, category: str, 
                                      start_date: Optional[date] = None, 
                                      end_date: Optional[date] = None) -> List[Expense]:
        """Получение расходов по категории"""
        try:
            query = """
                SELECT * FROM expenses 
                WHERE user_id = $1 AND category = $2
            """
            params = [user_id, category]
            param_count = 2
            
            if start_date:
                param_count += 1
                query += f" AND transaction_date >= ${param_count}"
                params.append(start_date)
            
            if end_date:
                param_count += 1
                query += f" AND transaction_date <= ${param_count}"
                params.append(end_date)
            
            query += " ORDER BY transaction_date DESC"
            
            results = await db_service.fetch_all(query, *params)
            return [Expense.from_dict(dict(row)) for row in results]
            
        except Exception as e:
            logger.error(f"Ошибка получения расходов по категории {category}: {e}")
            raise DatabaseError(f"Не удалось получить расходы по категории: {e}")
    
    async def get_expenses_summary(self, user_id: int, start_date: Optional[date] = None, 
                                  end_date: Optional[date] = None) -> Dict[str, Any]:
        """Получение сводки по расходам"""
        try:
            query = """
                SELECT 
                    category,
                    COUNT(*) as count,
                    SUM(amount) as total_amount,
                    AVG(amount) as avg_amount
                FROM expenses 
                WHERE user_id = $1
            """
            params = [user_id]
            param_count = 1
            
            if start_date:
                param_count += 1
                query += f" AND transaction_date >= ${param_count}"
                params.append(start_date)
            
            if end_date:
                param_count += 1
                query += f" AND transaction_date <= ${param_count}"
                params.append(end_date)
            
            query += " GROUP BY category ORDER BY total_amount DESC"
            
            results = await db_service.fetch_all(query, *params)
            
            summary = {
                'categories': [],
                'total_amount': Decimal('0'),
                'total_count': 0
            }
            
            for row in results:
                category_data = {
                    'category': row['category'],
                    'count': row['count'],
                    'total_amount': Decimal(str(row['total_amount'])),
                    'avg_amount': Decimal(str(row['avg_amount']))
                }
                summary['categories'].append(category_data)
                summary['total_amount'] += category_data['total_amount']
                summary['total_count'] += category_data['count']
            
            return summary
            
        except Exception as e:
            logger.error(f"Ошибка получения сводки расходов: {e}")
            raise DatabaseError(f"Не удалось получить сводку расходов: {e}")

# Глобальный экземпляр сервиса
expense_service = ExpenseService()
