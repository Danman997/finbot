"""
Обработчик расходов
"""
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import ContextTypes
from decimal import Decimal
from datetime import date
from typing import Optional
from handlers.base_handler import BaseHandler
from services.expense_service import expense_service
from services.classification_service import classification_service
from utils import logger, ValidationError, DatabaseError
from utils.validators import Validator

class ExpenseHandler(BaseHandler):
    """Обработчик расходов"""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Основной метод обработки"""
        if not await self.check_user_access(update, context):
            return
        
        try:
            text = update.message.text.strip()
            user_id = update.effective_user.id
            
            # Проверяем, является ли сообщение расходом
            if self._is_expense_input(text):
                await self._process_expense(update, context, text, user_id)
            else:
                await update.message.reply_text(
                    "❌ Неверный формат. Используйте: сумма описание",
                    reply_markup=self.get_main_menu_keyboard()
                )
                
        except Exception as e:
            await self.handle_error(update, context, e)
    
    def _is_expense_input(self, text: str) -> bool:
        """Проверка, является ли текст вводом расхода"""
        if not text:
            return False
        
        # Проверяем, начинается ли с числа
        parts = text.split(' ', 1)
        if len(parts) < 2:
            return False
        
        try:
            # Пытаемся преобразовать первую часть в число
            amount_str = parts[0].replace(',', '.')
            float(amount_str)
            return True
        except ValueError:
            return False
    
    async def _process_expense(self, update: Update, context: ContextTypes.DEFAULT_TYPE, 
                              text: str, user_id: int):
        """Обработка расхода"""
        try:
            # Парсим сумму и описание
            parts = text.split(' ', 1)
            amount_str = parts[0].replace(',', '.')
            description = parts[1].strip()
            
            # Валидируем данные
            amount = Validator.validate_amount(amount_str)
            description = Validator.validate_description(description)
            
            # Классифицируем расход
            category = classification_service.classify_expense(description)
            confidence = classification_service.get_classification_confidence(description)
            
            # Создаем расход
            expense = await expense_service.create_expense(
                user_id=user_id,
                amount=amount,
                description=description,
                category=category,
                transaction_date=date.today()
            )
            
            # Формируем ответ
            response = f"✅ Расход добавлен!\n\n"
            response += f"💰 Сумма: {self.format_amount(float(amount))}\n"
            response += f"📝 Описание: {description}\n"
            response += f"🏷️ Категория: {category}\n"
            
            if confidence < 0.7:
                response += f"⚠️ Категория определена автоматически (уверенность: {confidence:.1%})\n"
                response += f"💡 Если категория неверна, используйте '🔧 Исправить категории'"
            
            await update.message.reply_text(
                response,
                reply_markup=self.get_main_menu_keyboard()
            )
            
            logger.info(f"Добавлен расход: {description} ({amount} Тг) для пользователя {user_id}")
            
        except ValidationError as e:
            await update.message.reply_text(
                f"❌ {e.message}",
                reply_markup=self.get_main_menu_keyboard()
            )
        except DatabaseError as e:
            await update.message.reply_text(
                "❌ Ошибка сохранения расхода. Попробуйте позже.",
                reply_markup=self.get_main_menu_keyboard()
            )
        except Exception as e:
            await self.handle_error(update, context, e)
    
    async def get_expenses_summary(self, user_id: int, period: str = "month") -> str:
        """Получение сводки по расходам"""
        try:
            from datetime import datetime, timedelta
            
            # Определяем период
            end_date = date.today()
            if period == "today":
                start_date = end_date
            elif period == "week":
                start_date = end_date - timedelta(days=7)
            elif period == "month":
                start_date = end_date.replace(day=1)
            elif period == "year":
                start_date = end_date.replace(month=1, day=1)
            else:
                start_date = end_date.replace(day=1)
            
            # Получаем сводку
            summary = await expense_service.get_expenses_summary(user_id, start_date, end_date)
            
            if not summary['categories']:
                return f"📊 За {period} нет расходов"
            
            # Формируем ответ
            response = f"📊 Сводка расходов за {period}:\n\n"
            
            for category_data in summary['categories'][:10]:  # Показываем топ-10
                response += f"🏷️ {category_data['category']}: {self.format_amount(float(category_data['total_amount']))}\n"
            
            response += f"\n💰 Общая сумма: {self.format_amount(float(summary['total_amount']))}\n"
            response += f"📈 Количество транзакций: {summary['total_count']}"
            
            return response
            
        except Exception as e:
            logger.error(f"Ошибка получения сводки расходов: {e}")
            return "❌ Ошибка получения сводки расходов"
    
    async def get_expenses_by_category(self, user_id: int, category: str, limit: int = 10) -> str:
        """Получение расходов по категории"""
        try:
            from datetime import datetime, timedelta
            
            # Получаем расходы за последний месяц
            end_date = date.today()
            start_date = end_date.replace(day=1)
            
            expenses = await expense_service.get_expenses_by_category(
                user_id, category, start_date, end_date
            )
            
            if not expenses:
                return f"📊 Нет расходов по категории '{category}' за текущий месяц"
            
            # Формируем ответ
            response = f"📊 Расходы по категории '{category}':\n\n"
            
            for expense in expenses[:limit]:
                response += f"💰 {self.format_amount(float(expense.amount))} - {expense.description}\n"
                response += f"📅 {self.format_date(expense.transaction_date)}\n\n"
            
            if len(expenses) > limit:
                response += f"... и еще {len(expenses) - limit} расходов"
            
            return response
            
        except Exception as e:
            logger.error(f"Ошибка получения расходов по категории: {e}")
            return "❌ Ошибка получения расходов по категории"
    
    async def delete_expense(self, expense_id: int) -> bool:
        """Удаление расхода"""
        try:
            return await expense_service.delete_expense(expense_id)
        except Exception as e:
            logger.error(f"Ошибка удаления расхода {expense_id}: {e}")
            return False
    
    async def update_expense_category(self, expense_id: int, new_category: str) -> bool:
        """Обновление категории расхода"""
        try:
            expense = await expense_service.update_expense(expense_id, category=new_category)
            return expense is not None
        except Exception as e:
            logger.error(f"Ошибка обновления категории расхода {expense_id}: {e}")
            return False
