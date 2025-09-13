"""
Пример обновленных функций bot.py для работы с базой данных
Показывает, как заменить файловые операции на работу с БД
"""

# Импорты для работы с БД
from database import (
    get_user_by_telegram_id, create_user, get_user_categories, 
    add_expense, get_user_expenses, get_expenses_by_category,
    create_budget_plan, get_user_budget_plans, add_reminder, 
    get_user_reminders, get_user_setting, set_user_setting
)

# ============ ОБНОВЛЕННЫЕ ФУНКЦИИ ============

def get_user_info(user_id: int) -> dict:
    """
    Заменяет get_user_folder_path() и get_user_folder_info()
    Получает информацию о пользователе из БД
    """
    user = get_user_by_telegram_id(user_id)
    if not user:
        return None
    
    return {
        'id': user['id'],
        'telegram_id': user['telegram_id'],
        'username': user['username'],
        'folder_name': user['folder_name'],
        'role': user['role'],
        'created_at': user['created_at']
    }

def ensure_user_exists(user_id: int, username: str = None, folder_name: str = None) -> bool:
    """
    Заменяет create_user_folder()
    Создает пользователя в БД, если его нет
    """
    user = get_user_by_telegram_id(user_id)
    if not user:
        return create_user(user_id, username, folder_name)
    return True

def load_user_categories(user_id: int) -> list:
    """
    Заменяет загрузку из user_categories.json
    Получает категории пользователя из БД
    """
    categories = get_user_categories(user_id)
    return [
        {
            'name': cat['category_name'],
            'type': cat['category_type'],
            'color': cat['color'],
            'icon': cat['icon']
        }
        for cat in categories
    ]

def add_expense_to_database(user_id: int, category_name: str, amount: float, 
                           description: str = None, expense_date: date = None) -> bool:
    """
    Заменяет сохранение в expenses.json
    Добавляет расход в БД
    """
    # Получаем ID категории по имени
    categories = get_user_categories(user_id)
    category_id = None
    for cat in categories:
        if cat['category_name'] == category_name:
            category_id = cat['id']
            break
    
    if not category_id:
        return False
    
    return add_expense(user_id, category_id, amount, description, expense_date)

def get_expenses_for_period(user_id: int, start_date: date = None, end_date: date = None) -> list:
    """
    Заменяет загрузку из expenses.json
    Получает расходы за период из БД
    """
    expenses = get_user_expenses(user_id, start_date, end_date)
    return [
        {
            'amount': exp['amount'],
            'category': exp['category_name'],
            'description': exp['description'],
            'date': exp['date'].strftime('%Y-%m-%d'),
            'color': exp['color'],
            'icon': exp['icon']
        }
        for exp in expenses
    ]

def create_budget_plan_in_database(user_id: int, plan_data: dict) -> bool:
    """
    Заменяет сохранение в budget_plans.json
    Создает план бюджета в БД
    """
    from datetime import datetime
    
    return create_budget_plan(
        user_id,
        plan_data['name'],
        plan_data['total_amount'],
        datetime.strptime(plan_data['start_date'], '%Y-%m-%d').date(),
        datetime.strptime(plan_data['end_date'], '%Y-%m-%d').date(),
        plan_data.get('categories')
    )

def load_user_budget_plans(user_id: int) -> list:
    """
    Заменяет загрузку из budget_plans.json
    Получает планы бюджета из БД
    """
    plans = get_user_budget_plans(user_id)
    return [
        {
            'id': plan['id'],
            'name': plan['plan_name'],
            'total_amount': float(plan['total_amount']),
            'spent_amount': float(plan['spent_amount']),
            'start_date': plan['start_date'].strftime('%Y-%m-%d'),
            'end_date': plan['end_date'].strftime('%Y-%m-%d'),
            'categories': plan['categories'],
            'is_active': plan['is_active']
        }
        for plan in plans
    ]

def add_reminder_to_database(user_id: int, reminder_data: dict) -> bool:
    """
    Заменяет сохранение в reminders.json
    Добавляет напоминание в БД
    """
    from datetime import datetime
    
    return add_reminder(
        user_id,
        reminder_data['title'],
        reminder_data.get('description'),
        datetime.strptime(reminder_data['date'], '%Y-%m-%d').date(),
        datetime.strptime(reminder_data.get('time', '00:00'), '%H:%M').time(),
        reminder_data.get('recurring', False),
        reminder_data.get('pattern')
    )

def load_user_reminders(user_id: int) -> list:
    """
    Заменяет загрузку из reminders.json
    Получает напоминания из БД
    """
    reminders = get_user_reminders(user_id)
    return [
        {
            'id': rem['id'],
            'title': rem['title'],
            'description': rem['description'],
            'date': rem['reminder_date'].strftime('%Y-%m-%d'),
            'time': rem['reminder_time'].strftime('%H:%M') if rem['reminder_time'] else '00:00',
            'recurring': rem['is_recurring'],
            'pattern': rem['recurring_pattern'],
            'completed': rem['is_completed']
        }
        for rem in reminders
    ]

def get_user_config(user_id: int) -> dict:
    """
    Заменяет загрузку из user_config.json
    Получает настройки пользователя из БД
    """
    # Стандартные настройки
    default_settings = {
        'currency': 'RUB',
        'language': 'ru',
        'timezone': 'Europe/Moscow',
        'notifications': True,
        'daily_reminder': True,
        'week_start': 'monday'
    }
    
    # Загружаем настройки из БД
    user_settings = {}
    for key in default_settings.keys():
        value = get_user_setting(user_id, key)
        if value:
            # Преобразуем строковые значения в нужные типы
            if key in ['notifications', 'daily_reminder']:
                user_settings[key] = value.lower() == 'true'
            else:
                user_settings[key] = value
        else:
            user_settings[key] = default_settings[key]
    
    return user_settings

def save_user_config(user_id: int, config: dict) -> bool:
    """
    Заменяет сохранение в user_config.json
    Сохраняет настройки пользователя в БД
    """
    success = True
    for key, value in config.items():
        if not set_user_setting(user_id, key, str(value)):
            success = False
    
    return success

# ============ ПРИМЕР ОБНОВЛЕННОЙ КОМАНДЫ ============

async def start_command(update, context):
    """Обновленная команда /start для работы с БД"""
    user_id = update.effective_user.id
    username = update.effective_user.username
    
    # Убеждаемся, что пользователь существует в БД
    if not ensure_user_exists(user_id, username):
        await update.message.reply_text("❌ Ошибка создания профиля пользователя")
        return
    
    # Получаем информацию о пользователе
    user_info = get_user_info(user_id)
    if not user_info:
        await update.message.reply_text("❌ Пользователь не найден")
        return
    
    # Получаем настройки пользователя
    config = get_user_config(user_id)
    
    welcome_text = f"""
👋 Добро пожаловать в финансовый бот, {user_info['username'] or 'пользователь'}!

📊 Ваш профиль:
• ID: {user_info['telegram_id']}
• Валюта: {config['currency']}
• Язык: {config['language']}
• Уведомления: {'Включены' if config['notifications'] else 'Отключены'}

💡 Доступные команды:
/expenses - Управление расходами
/budget - Планы бюджета
/reminders - Напоминания
/settings - Настройки
/help - Справка
    """
    
    await update.message.reply_text(welcome_text)

# ============ ПРИМЕР ОБНОВЛЕННОЙ КОМАНДЫ РАСХОДОВ ============

async def add_expense_command(update, context):
    """Обновленная команда добавления расхода для работы с БД"""
    user_id = update.effective_user.id
    
    # Убеждаемся, что пользователь существует
    if not ensure_user_exists(user_id):
        await update.message.reply_text("❌ Пользователь не найден")
        return
    
    # Получаем категории пользователя
    categories = load_user_categories(user_id)
    
    if not categories:
        await update.message.reply_text("❌ Категории не найдены")
        return
    
    # Формируем клавиатуру с категориями
    keyboard = []
    for i in range(0, len(categories), 2):
        row = []
        for j in range(2):
            if i + j < len(categories):
                cat = categories[i + j]
                row.append(InlineKeyboardButton(
                    f"{cat['icon']} {cat['name']}", 
                    callback_data=f"expense_cat_{cat['name']}"
                ))
        keyboard.append(row)
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "💰 Выберите категорию для расхода:",
        reply_markup=reply_markup
    )

async def handle_expense_category(update, context):
    """Обработчик выбора категории расхода"""
    query = update.callback_query
    await query.answer()
    
    category_name = query.data.replace("expense_cat_", "")
    user_id = query.from_user.id
    
    # Сохраняем выбранную категорию в контексте
    context.user_data['selected_category'] = category_name
    
    await query.edit_message_text(
        f"💰 Категория: {category_name}\n"
        f"Введите сумму расхода:"
    )

# ============ КЛЮЧЕВЫЕ ИЗМЕНЕНИЯ ============

"""
ОСНОВНЫЕ ИЗМЕНЕНИЯ В КОДЕ:

1. Импорты:
   - Добавить: from database import *
   - Убрать: import json, os для работы с файлами

2. Функции пользователей:
   - get_user_folder_path() → get_user_by_telegram_id()
   - create_user_folder() → create_user()
   - get_user_folder_info() → get_user_info()

3. Функции категорий:
   - Загрузка из JSON → get_user_categories()
   - Сохранение в JSON → add_user_category()

4. Функции расходов:
   - Загрузка из expenses.json → get_user_expenses()
   - Сохранение в expenses.json → add_expense()

5. Функции планов:
   - Загрузка из budget_plans.json → get_user_budget_plans()
   - Сохранение в budget_plans.json → create_budget_plan()

6. Функции напоминаний:
   - Загрузка из reminders.json → get_user_reminders()
   - Сохранение в reminders.json → add_reminder()

7. Функции настроек:
   - Загрузка из user_config.json → get_user_config()
   - Сохранение в user_config.json → save_user_config()

ПРЕИМУЩЕСТВА:
- Нет необходимости в создании папок
- Автоматическое создание пользователей
- Надежное хранение данных
- Лучшая производительность
- Возможность сложных запросов
"""
