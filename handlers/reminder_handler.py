"""
Обработчик напоминаний
"""
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import ContextTypes
from decimal import Decimal
from datetime import date
from typing import Optional, List
from handlers.base_handler import BaseHandler
from services.reminder_service import reminder_service
from models.reminder import Reminder
from utils import logger, ValidationError, DatabaseError
from utils.validators import Validator

class ReminderHandler(BaseHandler):
    """Обработчик напоминаний"""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Основной метод обработки"""
        if not await self.check_user_access(update, context):
            return
        
        try:
            text = update.message.text.strip()
            user_id = update.effective_user.id
            
            # Обработка различных команд
            if text == "⏰ Напоминания":
                await self._show_reminders_menu(update, context, user_id)
            elif text == "➕ Добавить напоминание":
                await self._start_reminder_creation(update, context, user_id)
            elif text == "📋 Список напоминаний":
                await self._show_reminders_list(update, context, user_id)
            elif text == "✏️ Редактировать напоминание":
                await self._start_reminder_editing(update, context, user_id)
            elif text == "🗑️ Удалить напоминание":
                await self._start_reminder_deletion(update, context, user_id)
            else:
                await update.message.reply_text(
                    "❌ Неизвестная команда",
                    reply_markup=self.get_main_menu_keyboard()
                )
                
        except Exception as e:
            await self.handle_error(update, context, e)
    
    async def _show_reminders_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE, user_id: int):
        """Показ меню напоминаний"""
        try:
            # Получаем напоминания пользователя
            reminders = await reminder_service.get_user_reminders(user_id, active_only=True, limit=5)
            
            if not reminders:
                keyboard = [
                    [KeyboardButton("➕ Добавить напоминание")],
                    [KeyboardButton("🔙 Назад")]
                ]
                message = "⏰ Напоминания\n\nУ вас пока нет активных напоминаний."
            else:
                keyboard = [
                    [KeyboardButton("➕ Добавить напоминание"), KeyboardButton("📋 Список напоминаний")],
                    [KeyboardButton("✏️ Редактировать напоминание"), KeyboardButton("🗑️ Удалить напоминание")],
                    [KeyboardButton("🔙 Назад")]
                ]
                
                message = "⏰ Напоминания\n\n"
                message += "📋 Ваши напоминания:\n"
                for reminder in reminders[:3]:  # Показываем последние 3 напоминания
                    message += f"• {reminder.title}: {self.format_amount(float(reminder.amount))}\n"
                    message += f"  📅 {self.format_date(reminder.start_date)}\n"
                
                if len(reminders) > 3:
                    message += f"... и еще {len(reminders) - 3} напоминаний"
            
            await update.message.reply_text(
                message,
                reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
            )
            
        except Exception as e:
            await self.handle_error(update, context, e)
    
    async def _start_reminder_creation(self, update: Update, context: ContextTypes.DEFAULT_TYPE, user_id: int):
        """Начало создания напоминания"""
        try:
            context.user_data['reminder_user_id'] = user_id
            context.user_data['reminder_step'] = 'title'
            
            await update.message.reply_text(
                "⏰ Создание напоминания\n\n"
                "Введите название напоминания:",
                reply_markup=self.get_back_keyboard()
            )
            
        except Exception as e:
            await self.handle_error(update, context, e)
    
    async def _process_reminder_title(self, update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
        """Обработка ввода названия"""
        try:
            # Валидируем название
            title = Validator.validate_length(text, 1, 100, "Название напоминания")
            
            # Сохраняем данные
            context.user_data['reminder_title'] = title
            context.user_data['reminder_step'] = 'description'
            
            await update.message.reply_text(
                f"📝 Название: {title}\n\n"
                "Введите описание напоминания (или 'Пропустить'):",
                reply_markup=self.get_back_keyboard()
            )
            
        except ValidationError as e:
            await update.message.reply_text(
                f"❌ {e.message}",
                reply_markup=self.get_back_keyboard()
            )
        except Exception as e:
            await self.handle_error(update, context, e)
    
    async def _process_reminder_description(self, update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
        """Обработка ввода описания"""
        try:
            # Обрабатываем пропуск описания
            if text.lower() in ['пропустить', 'skip', '']:
                description = None
            else:
                description = Validator.validate_length(text, 1, 500, "Описание напоминания")
            
            # Сохраняем данные
            context.user_data['reminder_description'] = description
            context.user_data['reminder_step'] = 'amount'
            
            await update.message.reply_text(
                f"📄 Описание: {description or 'Не указано'}\n\n"
                "Введите сумму напоминания:",
                reply_markup=self.get_back_keyboard()
            )
            
        except ValidationError as e:
            await update.message.reply_text(
                f"❌ {e.message}",
                reply_markup=self.get_back_keyboard()
            )
        except Exception as e:
            await self.handle_error(update, context, e)
    
    async def _process_reminder_amount(self, update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
        """Обработка ввода суммы"""
        try:
            # Валидируем сумму
            amount = Validator.validate_amount(text)
            
            # Сохраняем данные
            context.user_data['reminder_amount'] = amount
            context.user_data['reminder_step'] = 'start_date'
            
            await update.message.reply_text(
                f"💰 Сумма: {self.format_amount(float(amount))}\n\n"
                "Введите дату начала в формате ДД.ММ.ГГГГ (например: 15.09.2025):",
                reply_markup=self.get_back_keyboard()
            )
            
        except ValidationError as e:
            await update.message.reply_text(
                f"❌ {e.message}",
                reply_markup=self.get_back_keyboard()
            )
        except Exception as e:
            await self.handle_error(update, context, e)
    
    async def _process_reminder_start_date(self, update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
        """Обработка ввода даты начала"""
        try:
            # Валидируем дату
            start_date = Validator.validate_date(text)
            
            # Сохраняем данные
            context.user_data['reminder_start_date'] = start_date
            context.user_data['reminder_step'] = 'end_date'
            
            await update.message.reply_text(
                f"📅 Дата начала: {self.format_date(start_date)}\n\n"
                "Введите дату окончания в формате ДД.ММ.ГГГГ (или 'Пропустить' для бессрочного):",
                reply_markup=self.get_back_keyboard()
            )
            
        except ValidationError as e:
            await update.message.reply_text(
                f"❌ {e.message}",
                reply_markup=self.get_back_keyboard()
            )
        except Exception as e:
            await self.handle_error(update, context, e)
    
    async def _process_reminder_end_date(self, update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
        """Обработка ввода даты окончания"""
        try:
            # Обрабатываем пропуск даты окончания
            if text.lower() in ['пропустить', 'skip', '']:
                end_date = None
            else:
                end_date = Validator.validate_date(text)
                
                # Проверяем, что дата окончания не раньше даты начала
                start_date = context.user_data['reminder_start_date']
                if end_date < start_date:
                    raise ValidationError("Дата окончания не может быть раньше даты начала")
            
            # Создаем напоминание
            await self._complete_reminder_creation(update, context, end_date)
            
        except ValidationError as e:
            await update.message.reply_text(
                f"❌ {e.message}",
                reply_markup=self.get_back_keyboard()
            )
        except Exception as e:
            await self.handle_error(update, context, e)
    
    async def _complete_reminder_creation(self, update: Update, context: ContextTypes.DEFAULT_TYPE, end_date: Optional[date]):
        """Завершение создания напоминания"""
        try:
            user_id = context.user_data['reminder_user_id']
            title = context.user_data['reminder_title']
            description = context.user_data.get('reminder_description')
            amount = context.user_data['reminder_amount']
            start_date = context.user_data['reminder_start_date']
            
            # Создаем напоминание
            reminder = await reminder_service.create_reminder(
                user_id=user_id,
                title=title,
                description=description,
                amount=amount,
                start_date=start_date,
                end_date=end_date
            )
            
            # Формируем ответ
            response = f"✅ Напоминание создано!\n\n"
            response += f"📝 Название: {title}\n"
            response += f"📄 Описание: {description or 'Не указано'}\n"
            response += f"💰 Сумма: {self.format_amount(float(amount))}\n"
            response += f"📅 Дата начала: {self.format_date(start_date)}\n"
            response += f"📅 Дата окончания: {self.format_date(end_date) if end_date else 'Бессрочно'}\n"
            
            await update.message.reply_text(
                response,
                reply_markup=self.get_main_menu_keyboard()
            )
            
            # Очищаем данные
            self._clear_reminder_data(context)
            
        except Exception as e:
            await self.handle_error(update, context, e)
    
    async def _show_reminders_list(self, update: Update, context: ContextTypes.DEFAULT_TYPE, user_id: int):
        """Показ списка напоминаний"""
        try:
            reminders = await reminder_service.get_user_reminders(user_id, active_only=True)
            
            if not reminders:
                await update.message.reply_text(
                    "📋 У вас пока нет активных напоминаний.",
                    reply_markup=self.get_back_keyboard()
                )
                return
            
            # Формируем список
            response = "📋 Ваши напоминания:\n\n"
            
            for i, reminder in enumerate(reminders, 1):
                response += f"{i}. {reminder.title}\n"
                response += f"   💰 {self.format_amount(float(reminder.amount))}\n"
                response += f"   📅 {self.format_date(reminder.start_date)}"
                if reminder.end_date:
                    response += f" - {self.format_date(reminder.end_date)}"
                response += "\n\n"
            
            keyboard = [
                [KeyboardButton("✏️ Редактировать напоминание"), KeyboardButton("🗑️ Удалить напоминание")],
                [KeyboardButton("🔙 Назад")]
            ]
            
            await update.message.reply_text(
                response,
                reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
            )
            
        except Exception as e:
            await self.handle_error(update, context, e)
    
    def _clear_reminder_data(self, context: ContextTypes.DEFAULT_TYPE):
        """Очистка данных напоминания"""
        keys_to_remove = [
            'reminder_user_id', 'reminder_step', 'reminder_title', 
            'reminder_description', 'reminder_amount', 'reminder_start_date'
        ]
        for key in keys_to_remove:
            context.user_data.pop(key, None)
