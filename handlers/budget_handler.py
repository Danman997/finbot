"""
Обработчик планирования бюджета
"""
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import ContextTypes
from decimal import Decimal
from datetime import date
from typing import Optional, List
from handlers.base_handler import BaseHandler
from services.budget_service import budget_service
from models.budget_plan import BudgetPlan, BudgetPlanItem
from utils import logger, ValidationError, DatabaseError
from utils.validators import Validator

class BudgetHandler(BaseHandler):
    """Обработчик планирования бюджета"""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Основной метод обработки"""
        if not await self.check_user_access(update, context):
            return
        
        try:
            text = update.message.text.strip()
            user_id = update.effective_user.id
            
            # Обработка различных команд
            if text == "📅 Планирование":
                await self._show_planning_menu(update, context, user_id)
            elif text == "➕ Добавить планирование":
                await self._start_planning(update, context, user_id)
            elif text == "📋 Список планов":
                await self._show_plans_list(update, context, user_id)
            elif text == "✏️ Редактировать план":
                await self._start_plan_editing(update, context, user_id)
            elif text == "🗑️ Удалить план":
                await self._start_plan_deletion(update, context, user_id)
            else:
                await update.message.reply_text(
                    "❌ Неизвестная команда",
                    reply_markup=self.get_main_menu_keyboard()
                )
                
        except Exception as e:
            await self.handle_error(update, context, e)
    
    async def _show_planning_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE, user_id: int):
        """Показ меню планирования"""
        try:
            # Получаем планы пользователя
            plans = await budget_service.get_user_budget_plans(user_id, limit=5)
            
            if not plans:
                keyboard = [
                    [KeyboardButton("➕ Добавить планирование")],
                    [KeyboardButton("🔙 Назад")]
                ]
                message = "📅 Планирование бюджета\n\nУ вас пока нет планов бюджета."
            else:
                keyboard = [
                    [KeyboardButton("➕ Добавить планирование"), KeyboardButton("📋 Список планов")],
                    [KeyboardButton("✏️ Редактировать план"), KeyboardButton("🗑️ Удалить план")],
                    [KeyboardButton("🔙 Назад")]
                ]
                
                message = "📅 Планирование бюджета\n\n"
                message += "📋 Ваши планы:\n"
                for plan in plans[:3]:  # Показываем последние 3 плана
                    message += f"• {self.format_month_year(plan.plan_month)}: {self.format_amount(float(plan.total_amount))}\n"
                
                if len(plans) > 3:
                    message += f"... и еще {len(plans) - 3} планов"
            
            await update.message.reply_text(
                message,
                reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
            )
            
        except Exception as e:
            await self.handle_error(update, context, e)
    
    async def _start_planning(self, update: Update, context: ContextTypes.DEFAULT_TYPE, user_id: int):
        """Начало создания плана"""
        try:
            context.user_data['planning_user_id'] = user_id
            context.user_data['planning_step'] = 'month'
            
            await update.message.reply_text(
                "📅 Создание плана бюджета\n\n"
                "Введите месяц и год в формате ММ.ГГГГ (например: 09.2025):",
                reply_markup=self.get_back_keyboard()
            )
            
        except Exception as e:
            await self.handle_error(update, context, e)
    
    async def _process_planning_month(self, update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
        """Обработка ввода месяца"""
        try:
            # Валидируем месяц
            month, year = Validator.validate_month_year(text)
            plan_date = date(year, month, 1)
            
            # Проверяем, нет ли уже плана на этот месяц
            existing_plan = await budget_service.get_budget_plan_by_month(
                context.user_data['planning_user_id'], month, year
            )
            
            if existing_plan:
                await update.message.reply_text(
                    f"❌ План на {text} уже существует.\n"
                    f"Выберите другой месяц или отредактируйте существующий план.",
                    reply_markup=self.get_back_keyboard()
                )
                return
            
            # Сохраняем данные
            context.user_data['planning_month'] = plan_date
            context.user_data['planning_step'] = 'total'
            
            await update.message.reply_text(
                f"📅 Месяц: {text}\n\n"
                "Введите общую сумму плана:",
                reply_markup=self.get_back_keyboard()
            )
            
        except ValidationError as e:
            await update.message.reply_text(
                f"❌ {e.message}",
                reply_markup=self.get_back_keyboard()
            )
        except Exception as e:
            await self.handle_error(update, context, e)
    
    async def _process_planning_total(self, update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
        """Обработка ввода общей суммы"""
        try:
            # Валидируем сумму
            total_amount = Validator.validate_amount(text)
            
            # Сохраняем данные
            context.user_data['planning_total'] = total_amount
            context.user_data['planning_step'] = 'categories'
            context.user_data['planning_items'] = []
            
            await update.message.reply_text(
                f"💰 Общая сумма: {self.format_amount(float(total_amount))}\n\n"
                "Теперь добавьте категории. Введите название категории:",
                reply_markup=self.get_back_keyboard()
            )
            
        except ValidationError as e:
            await update.message.reply_text(
                f"❌ {e.message}",
                reply_markup=self.get_back_keyboard()
            )
        except Exception as e:
            await self.handle_error(update, context, e)
    
    async def _process_planning_category(self, update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
        """Обработка ввода категории"""
        try:
            # Валидируем категорию
            category = Validator.validate_category(text)
            
            # Сохраняем категорию
            context.user_data['current_category'] = category
            context.user_data['planning_step'] = 'amount'
            
            await update.message.reply_text(
                f"🏷️ Категория: {category}\n\n"
                "Введите сумму для этой категории:",
                reply_markup=self.get_back_keyboard()
            )
            
        except ValidationError as e:
            await update.message.reply_text(
                f"❌ {e.message}",
                reply_markup=self.get_back_keyboard()
            )
        except Exception as e:
            await self.handle_error(update, context, e)
    
    async def _process_planning_amount(self, update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
        """Обработка ввода суммы категории"""
        try:
            # Валидируем сумму
            amount = Validator.validate_amount(text)
            
            # Создаем элемент плана
            item = BudgetPlanItem(
                category=context.user_data['current_category'],
                amount=amount
            )
            
            # Добавляем в список
            context.user_data['planning_items'].append(item)
            
            # Показываем текущий прогресс
            total_planned = sum(item.amount for item in context.user_data['planning_items'])
            remaining = context.user_data['planning_total'] - total_planned
            
            if remaining <= 0:
                # План завершен
                await self._complete_planning(update, context)
            else:
                # Продолжаем добавление категорий
                await update.message.reply_text(
                    f"✅ Категория добавлена: {item.category} - {self.format_amount(float(amount))}\n\n"
                    f"📊 Прогресс: {self.format_amount(float(total_planned))} / {self.format_amount(float(context.user_data['planning_total']))}\n"
                    f"💰 Осталось: {self.format_amount(float(remaining))}\n\n"
                    "Введите название следующей категории или 'Готово' для завершения:",
                    reply_markup=self.get_back_keyboard()
                )
            
        except ValidationError as e:
            await update.message.reply_text(
                f"❌ {e.message}",
                reply_markup=self.get_back_keyboard()
            )
        except Exception as e:
            await self.handle_error(update, context, e)
    
    async def _complete_planning(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Завершение создания плана"""
        try:
            user_id = context.user_data['planning_user_id']
            plan_month = context.user_data['planning_month']
            total_amount = context.user_data['planning_total']
            items = context.user_data['planning_items']
            
            # Создаем план
            plan = await budget_service.create_budget_plan(
                user_id=user_id,
                plan_month=plan_month,
                total_amount=total_amount,
                items=items
            )
            
            # Формируем ответ
            response = f"✅ План бюджета создан!\n\n"
            response += f"📅 Месяц: {self.format_month_year(plan_month)}\n"
            response += f"💰 Общая сумма: {self.format_amount(float(total_amount))}\n\n"
            response += f"📋 Категории:\n"
            
            for item in items:
                response += f"• {item.category}: {self.format_amount(float(item.amount))}\n"
            
            await update.message.reply_text(
                response,
                reply_markup=self.get_main_menu_keyboard()
            )
            
            # Очищаем данные
            self._clear_planning_data(context)
            
        except Exception as e:
            await self.handle_error(update, context, e)
    
    async def _show_plans_list(self, update: Update, context: ContextTypes.DEFAULT_TYPE, user_id: int):
        """Показ списка планов"""
        try:
            plans = await budget_service.get_user_budget_plans(user_id)
            
            if not plans:
                await update.message.reply_text(
                    "📋 У вас пока нет планов бюджета.",
                    reply_markup=self.get_back_keyboard()
                )
                return
            
            # Формируем список
            response = "📋 Ваши планы бюджета:\n\n"
            
            for i, plan in enumerate(plans, 1):
                response += f"{i}. {self.format_month_year(plan.plan_month)}\n"
                response += f"   💰 {self.format_amount(float(plan.total_amount))}\n"
                response += f"   📊 {len(plan.items)} категорий\n\n"
            
            keyboard = [
                [KeyboardButton("✏️ Редактировать план"), KeyboardButton("🗑️ Удалить план")],
                [KeyboardButton("🔙 Назад")]
            ]
            
            await update.message.reply_text(
                response,
                reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
            )
            
        except Exception as e:
            await self.handle_error(update, context, e)
    
    def _clear_planning_data(self, context: ContextTypes.DEFAULT_TYPE):
        """Очистка данных планирования"""
        keys_to_remove = [
            'planning_user_id', 'planning_step', 'planning_month', 
            'planning_total', 'planning_items', 'current_category'
        ]
        for key in keys_to_remove:
            context.user_data.pop(key, None)
