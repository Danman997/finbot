"""
Базовый обработчик
"""
from abc import ABC, abstractmethod
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import ContextTypes
from typing import Optional, List, Dict, Any
from utils import logger, check_rate_limit, ValidationError, DatabaseError, RateLimitError

class BaseHandler(ABC):
    """Базовый класс для всех обработчиков"""
    
    def __init__(self):
        self.logger = logger
    
    async def handle_error(self, update: Update, context: ContextTypes.DEFAULT_TYPE, error: Exception):
        """Обработка ошибок"""
        user_id = update.effective_user.id if update.effective_user else 0
        
        if isinstance(error, RateLimitError):
            await update.message.reply_text(
                f"⏰ {error.message}",
                reply_markup=self.get_main_menu_keyboard()
            )
        elif isinstance(error, ValidationError):
            await update.message.reply_text(
                f"❌ {error.message}",
                reply_markup=self.get_main_menu_keyboard()
            )
        elif isinstance(error, DatabaseError):
            await update.message.reply_text(
                "❌ Ошибка базы данных. Попробуйте позже.",
                reply_markup=self.get_main_menu_keyboard()
            )
        else:
            await update.message.reply_text(
                "❌ Произошла ошибка. Попробуйте позже.",
                reply_markup=self.get_main_menu_keyboard()
            )
        
        self.logger.error(f"Ошибка для пользователя {user_id}: {error}")
    
    async def check_user_access(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
        """Проверка доступа пользователя"""
        try:
            user_id = update.effective_user.id
            check_rate_limit(user_id)
            return True
        except RateLimitError as e:
            await self.handle_error(update, context, e)
            return False
        except Exception as e:
            await self.handle_error(update, context, e)
            return False
    
    def get_main_menu_keyboard(self) -> ReplyKeyboardMarkup:
        """Получение клавиатуры главного меню"""
        keyboard = [
            [KeyboardButton("💸 Добавить расход"), KeyboardButton("📊 Отчеты")],
            [KeyboardButton("📅 Планирование"), KeyboardButton("⏰ Напоминания")],
            [KeyboardButton("📈 Аналитика"), KeyboardButton("🔧 Исправить категории")],
            [KeyboardButton("👥 Группы"), KeyboardButton("ℹ️ Помощь")]
        ]
        return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    def get_back_keyboard(self) -> ReplyKeyboardMarkup:
        """Получение клавиатуры с кнопкой назад"""
        keyboard = [[KeyboardButton("🔙 Назад")]]
        return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    def get_yes_no_keyboard(self) -> ReplyKeyboardMarkup:
        """Получение клавиатуры да/нет"""
        keyboard = [
            [KeyboardButton("✅ Да"), KeyboardButton("❌ Нет")],
            [KeyboardButton("🔙 Назад")]
        ]
        return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    def get_period_keyboard(self) -> ReplyKeyboardMarkup:
        """Получение клавиатуры выбора периода"""
        keyboard = [
            [KeyboardButton("Сегодня"), KeyboardButton("Неделя")],
            [KeyboardButton("Месяц"), KeyboardButton("Год")],
            [KeyboardButton("🔙 Назад")]
        ]
        return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    def get_admin_menu_keyboard(self) -> ReplyKeyboardMarkup:
        """Получение клавиатуры админ-меню"""
        keyboard = [
            [KeyboardButton("👥 Добавить пользователя"), KeyboardButton("📋 Список пользователей")],
            [KeyboardButton("📁 Управление папками"), KeyboardButton("🔧 Роли пользователей")],
            [KeyboardButton("📊 Статистика системы"), KeyboardButton("🔙 Главное меню")]
        ]
        return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    def format_amount(self, amount: float) -> str:
        """Форматирование суммы"""
        return f"{amount:,.2f} Тг"
    
    def format_date(self, date_obj) -> str:
        """Форматирование даты"""
        if hasattr(date_obj, 'strftime'):
            return date_obj.strftime("%d.%m.%Y")
        return str(date_obj)
    
    def format_month_year(self, date_obj) -> str:
        """Форматирование месяца и года"""
        if hasattr(date_obj, 'strftime'):
            return date_obj.strftime("%m.%Y")
        return str(date_obj)
    
    def create_keyboard_from_list(self, items: List[str], columns: int = 2) -> ReplyKeyboardMarkup:
        """Создание клавиатуры из списка элементов"""
        keyboard = []
        for i in range(0, len(items), columns):
            row = items[i:i + columns]
            keyboard.append([KeyboardButton(item) for item in row])
        return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    def create_numbered_keyboard(self, items: List[str], columns: int = 1) -> ReplyKeyboardMarkup:
        """Создание нумерованной клавиатуры"""
        keyboard = []
        for i, item in enumerate(items, 1):
            keyboard.append([KeyboardButton(f"{i}. {item}")])
        return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    @abstractmethod
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Основной метод обработки"""
        pass
