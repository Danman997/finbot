"""
Сервис для работы с планированием бюджета
"""
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from decimal import Decimal
from models.budget_plan import BudgetPlan, BudgetPlanItem
from services.database_service import db_service
from utils import logger, DatabaseError, ValidationError

class BudgetService:
    """Сервис для работы с планированием бюджета"""
    
    async def create_budget_plan(self, user_id: int, plan_month: date, 
                                total_amount: Decimal, items: List[BudgetPlanItem]) -> BudgetPlan:
        """Создание нового плана бюджета"""
        try:
            async with db_service.transaction() as conn:
                # Создаем план
                plan_query = """
                    INSERT INTO budget_plans (user_id, plan_month, total_amount, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5)
                    RETURNING *
                """
                
                now = datetime.now()
                plan_result = await conn.fetchrow(
                    plan_query, user_id, plan_month, total_amount, now, now
                )
                
                if not plan_result:
                    raise DatabaseError("Не удалось создать план бюджета")
                
                plan_id = plan_result['id']
                
                # Создаем элементы плана
                for item in items:
                    item_query = """
                        INSERT INTO budget_plan_items (plan_id, category, amount, comment)
                        VALUES ($1, $2, $3, $4)
                    """
                    await conn.execute(item_query, plan_id, item.category, item.amount, item.comment)
                
                # Получаем созданный план с элементами
                created_plan = await self.get_budget_plan(plan_id)
                if not created_plan:
                    raise DatabaseError("Не удалось получить созданный план")
                
                logger.info(f"Создан план бюджета: {plan_month.strftime('%m.%Y')} для пользователя {user_id}")
                return created_plan
                
        except Exception as e:
            logger.error(f"Ошибка создания плана бюджета: {e}")
            raise DatabaseError(f"Не удалось создать план бюджета: {e}")
    
    async def get_budget_plan(self, plan_id: int) -> Optional[BudgetPlan]:
        """Получение плана бюджета по ID"""
        try:
            # Получаем план
            plan_query = "SELECT * FROM budget_plans WHERE id = $1"
            plan_result = await db_service.fetch_one(plan_query, plan_id)
            
            if not plan_result:
                return None
            
            # Получаем элементы плана
            items_query = "SELECT * FROM budget_plan_items WHERE plan_id = $1 ORDER BY id"
            items_results = await db_service.fetch_all(items_query, plan_id)
            
            plan = BudgetPlan.from_dict(dict(plan_result))
            plan.items = [BudgetPlanItem.from_dict(dict(row)) for row in items_results]
            
            return plan
            
        except Exception as e:
            logger.error(f"Ошибка получения плана бюджета {plan_id}: {e}")
            raise DatabaseError(f"Не удалось получить план бюджета: {e}")
    
    async def get_user_budget_plans(self, user_id: int, limit: int = 100, 
                                   offset: int = 0) -> List[BudgetPlan]:
        """Получение планов бюджета пользователя"""
        try:
            query = """
                SELECT * FROM budget_plans 
                WHERE user_id = $1
                ORDER BY plan_month DESC
                LIMIT $2 OFFSET $3
            """
            results = await db_service.fetch_all(query, user_id, limit, offset)
            
            plans = []
            for row in results:
                plan = BudgetPlan.from_dict(dict(row))
                # Получаем элементы для каждого плана
                items_query = "SELECT * FROM budget_plan_items WHERE plan_id = $1 ORDER BY id"
                items_results = await db_service.fetch_all(items_query, plan.id)
                plan.items = [BudgetPlanItem.from_dict(dict(item_row)) for item_row in items_results]
                plans.append(plan)
            
            return plans
            
        except Exception as e:
            logger.error(f"Ошибка получения планов бюджета пользователя {user_id}: {e}")
            raise DatabaseError(f"Не удалось получить планы бюджета: {e}")
    
    async def get_budget_plan_by_month(self, user_id: int, month: int, year: int) -> Optional[BudgetPlan]:
        """Получение плана бюджета по месяцу и году"""
        try:
            query = """
                SELECT * FROM budget_plans 
                WHERE user_id = $1 AND EXTRACT(MONTH FROM plan_month) = $2 AND EXTRACT(YEAR FROM plan_month) = $3
            """
            result = await db_service.fetch_one(query, user_id, month, year)
            
            if result:
                plan = BudgetPlan.from_dict(dict(result))
                # Получаем элементы плана
                items_query = "SELECT * FROM budget_plan_items WHERE plan_id = $1 ORDER BY id"
                items_results = await db_service.fetch_all(items_query, plan.id)
                plan.items = [BudgetPlanItem.from_dict(dict(row)) for row in items_results]
                return plan
            
            return None
            
        except Exception as e:
            logger.error(f"Ошибка получения плана бюджета по месяцу {month}.{year}: {e}")
            raise DatabaseError(f"Не удалось получить план бюджета: {e}")
    
    async def update_budget_plan(self, plan_id: int, **kwargs) -> Optional[BudgetPlan]:
        """Обновление плана бюджета"""
        try:
            async with db_service.transaction() as conn:
                # Формируем SET часть запроса
                set_parts = []
                values = []
                param_count = 1
                
                for key, value in kwargs.items():
                    if key in ['plan_month', 'total_amount']:
                        set_parts.append(f"{key} = ${param_count}")
                        values.append(value)
                        param_count += 1
                
                if not set_parts:
                    raise ValidationError("Нет полей для обновления")
                
                # Добавляем updated_at
                set_parts.append(f"updated_at = ${param_count}")
                values.append(datetime.now())
                param_count += 1
                
                # Добавляем plan_id в конец
                values.append(plan_id)
                
                query = f"""
                    UPDATE budget_plans 
                    SET {', '.join(set_parts)}
                    WHERE id = ${param_count}
                    RETURNING *
                """
                
                result = await conn.fetchrow(query, *values)
                
                if result:
                    logger.info(f"Обновлен план бюджета: {plan_id}")
                    return await self.get_budget_plan(plan_id)
                return None
                
        except Exception as e:
            logger.error(f"Ошибка обновления плана бюджета {plan_id}: {e}")
            raise DatabaseError(f"Не удалось обновить план бюджета: {e}")
    
    async def update_budget_plan_items(self, plan_id: int, items: List[BudgetPlanItem]) -> Optional[BudgetPlan]:
        """Обновление элементов плана бюджета"""
        try:
            async with db_service.transaction() as conn:
                # Удаляем старые элементы
                delete_query = "DELETE FROM budget_plan_items WHERE plan_id = $1"
                await conn.execute(delete_query, plan_id)
                
                # Добавляем новые элементы
                for item in items:
                    insert_query = """
                        INSERT INTO budget_plan_items (plan_id, category, amount, comment)
                        VALUES ($1, $2, $3, $4)
                    """
                    await conn.execute(insert_query, plan_id, item.category, item.amount, item.comment)
                
                # Обновляем общую сумму
                total_amount = sum(item.amount for item in items)
                update_query = """
                    UPDATE budget_plans 
                    SET total_amount = $1, updated_at = $2
                    WHERE id = $3
                """
                await conn.execute(update_query, total_amount, datetime.now(), plan_id)
                
                logger.info(f"Обновлены элементы плана бюджета: {plan_id}")
                return await self.get_budget_plan(plan_id)
                
        except Exception as e:
            logger.error(f"Ошибка обновления элементов плана бюджета {plan_id}: {e}")
            raise DatabaseError(f"Не удалось обновить элементы плана бюджета: {e}")
    
    async def delete_budget_plan(self, plan_id: int) -> bool:
        """Удаление плана бюджета"""
        try:
            async with db_service.transaction() as conn:
                # Удаляем элементы плана
                delete_items_query = "DELETE FROM budget_plan_items WHERE plan_id = $1"
                await conn.execute(delete_items_query, plan_id)
                
                # Удаляем план
                delete_plan_query = "DELETE FROM budget_plans WHERE id = $1"
                result = await conn.execute(delete_plan_query, plan_id)
                
                if "DELETE 1" in result:
                    logger.info(f"Удален план бюджета: {plan_id}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Ошибка удаления плана бюджета {plan_id}: {e}")
            raise DatabaseError(f"Не удалось удалить план бюджета: {e}")
    
    async def get_budget_summary(self, user_id: int, year: int) -> Dict[str, Any]:
        """Получение сводки по планам бюджета за год"""
        try:
            query = """
                SELECT 
                    EXTRACT(MONTH FROM plan_month) as month,
                    COUNT(*) as plans_count,
                    SUM(total_amount) as total_amount
                FROM budget_plans 
                WHERE user_id = $1 AND EXTRACT(YEAR FROM plan_month) = $2
                GROUP BY EXTRACT(MONTH FROM plan_month)
                ORDER BY month
            """
            results = await db_service.fetch_all(query, user_id, year)
            
            summary = {
                'year': year,
                'months': [],
                'total_plans': 0,
                'total_amount': Decimal('0')
            }
            
            for row in results:
                month_data = {
                    'month': int(row['month']),
                    'plans_count': row['plans_count'],
                    'total_amount': Decimal(str(row['total_amount']))
                }
                summary['months'].append(month_data)
                summary['total_plans'] += month_data['plans_count']
                summary['total_amount'] += month_data['total_amount']
            
            return summary
            
        except Exception as e:
            logger.error(f"Ошибка получения сводки планов бюджета: {e}")
            raise DatabaseError(f"Не удалось получить сводку планов бюджета: {e}")

# Глобальный экземпляр сервиса
budget_service = BudgetService()
