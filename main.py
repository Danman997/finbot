"""
Главный файл приложения FinBot
"""
import asyncio
import sys
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from config import config
from utils import logger, setup_logger
from services.database_service import db_service
from services.classification_service import classification_service
from handlers import (
    ExpenseHandler, BudgetHandler, ReminderHandler, 
    AnalyticsHandler, AdminHandler, GroupHandler
)

# Инициализация обработчиков
expense_handler = ExpenseHandler()
budget_handler = BudgetHandler()
reminder_handler = ReminderHandler()
analytics_handler = AnalyticsHandler()
admin_handler = AdminHandler()
group_handler = GroupHandler()

async def start_command(update, context):
    """Обработчик команды /start"""
    try:
        user_id = update.effective_user.id
        username = update.effective_user.username or update.effective_user.first_name
        
        # Проверяем, существует ли пользователь
        from services.user_service import user_service
        user = await user_service.get_user(user_id)
        
        if not user:
            # Создаем нового пользователя
            user = await user_service.create_user(
                user_id=user_id,
                username=username
            )
            logger.info(f"Создан новый пользователь: {username} (ID: {user_id})")
        
        # Обновляем последнюю активность
        await user_service.update_user(user_id, last_activity=asyncio.get_event_loop().time())
        
        welcome_message = f"""
🤖 Добро пожаловать в FinBot, {username}!

Я помогу вам:
💸 Учитывать расходы
📅 Планировать бюджет
⏰ Напоминать о платежах
📊 Анализировать траты

Выберите действие из меню ниже:
        """
        
        await update.message.reply_text(
            welcome_message,
            reply_markup=expense_handler.get_main_menu_keyboard()
        )
        
    except Exception as e:
        logger.error(f"Ошибка в команде /start: {e}")
        await update.message.reply_text(
            "❌ Произошла ошибка. Попробуйте позже.",
            reply_markup=expense_handler.get_main_menu_keyboard()
        )

async def help_command(update, context):
    """Обработчик команды /help"""
    help_text = """
🤖 FinBot - Помощник по управлению финансами

📋 Основные команды:
/start - Начать работу с ботом
/help - Показать эту справку

💸 Добавление расходов:
Просто отправьте сообщение в формате: "сумма описание"
Например: "1500 кофе"

📅 Планирование бюджета:
• Создавайте планы по месяцам
• Разбивайте на категории
• Следите за выполнением

⏰ Напоминания:
• Создавайте напоминания о платежах
• Устанавливайте даты
• Получайте уведомления

📊 Аналитика:
• Просматривайте отчеты
• Сравнивайте с планами
• Анализируйте тренды

🔧 Исправление категорий:
• Корректируйте автоматическую классификацию
• Улучшайте точность бота

👥 Группы:
• Создавайте группы пользователей
• Совместно планируйте бюджет
• Отслеживайте общие расходы

❓ Нужна помощь? Обратитесь к администратору.
    """
    
    await update.message.reply_text(help_text)

async def error_handler(update, context):
    """Обработчик ошибок"""
    logger.error(f"Ошибка: {context.error}")
    
    if update and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "❌ Произошла ошибка. Попробуйте позже.",
                reply_markup=expense_handler.get_main_menu_keyboard()
            )
        except:
            pass

async def message_handler(update, context):
    """Основной обработчик сообщений"""
    try:
        text = update.message.text.strip()
        
        # Обработка команд
        if text == "💸 Добавить расход":
            await expense_handler.handle(update, context)
        elif text == "📅 Планирование":
            await budget_handler.handle(update, context)
        elif text == "⏰ Напоминания":
            await reminder_handler.handle(update, context)
        elif text == "📈 Аналитика":
            await analytics_handler.handle(update, context)
        elif text == "👥 Группы":
            await group_handler.handle(update, context)
        elif text in ["👥 Добавить пользователя", "📋 Список пользователей", 
                     "📁 Управление папками", "🔧 Роли пользователей", 
                     "📊 Статистика системы"]:
            await admin_handler.handle(update, context)
        elif text == "🔙 Назад":
            await start_command(update, context)
        elif text == "ℹ️ Помощь":
            await help_command(update, context)
        else:
            # Проверяем, является ли сообщение расходом
            if expense_handler._is_expense_input(text):
                await expense_handler.handle(update, context)
            else:
                await update.message.reply_text(
                    "❌ Неизвестная команда. Используйте меню или /help для справки.",
                    reply_markup=expense_handler.get_main_menu_keyboard()
                )
                
    except Exception as e:
        logger.error(f"Ошибка в обработчике сообщений: {e}")
        await error_handler(update, context)

async def initialize_services():
    """Инициализация сервисов"""
    try:
        logger.info("Инициализация сервисов...")
        
        # Инициализация базы данных
        await db_service.initialize()
        logger.info("✅ База данных инициализирована")
        
        # Инициализация классификации
        classification_service.train_model()
        logger.info("✅ Сервис классификации инициализирован")
        
        logger.info("✅ Все сервисы инициализированы")
        
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации сервисов: {e}")
        raise

async def cleanup_services():
    """Очистка ресурсов"""
    try:
        logger.info("Очистка ресурсов...")
        await db_service.close()
        logger.info("✅ Ресурсы очищены")
    except Exception as e:
        logger.error(f"Ошибка очистки ресурсов: {e}")

def main():
    """Главная функция"""
    try:
        # Настройка логирования
        setup_logger("finbot")
        logger.info("🚀 Запуск FinBot...")
        
        # Создание приложения
        application = Application.builder().token(config.bot.token).build()
        
        # Добавление обработчиков
        application.add_handler(CommandHandler("start", start_command))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))
        
        # Обработчик ошибок
        application.add_error_handler(error_handler)
        
        logger.info("✅ Обработчики добавлены")
        
        # Инициализация сервисов
        asyncio.run(initialize_services())
        
        # Запуск бота
        logger.info("🤖 FinBot запущен и готов к работе!")
        application.run_polling(
            allowed_updates=["message", "callback_query"],
            drop_pending_updates=True
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 Получен сигнал остановки")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        sys.exit(1)
    finally:
        # Очистка ресурсов
        asyncio.run(cleanup_services())
        logger.info("👋 FinBot остановлен")

if __name__ == "__main__":
    main()
