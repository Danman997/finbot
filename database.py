"""
Модуль для работы с базой данных PostgreSQL
Содержит все функции для работы с пользователями и их данными
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Класс для управления подключением к базе данных"""
    
    def __init__(self):
        """Инициализация подключения к базе данных"""
        self.connection = None
        # Не подключаемся сразу, чтобы не падать при импорте
        # self.connect()
    
    def connect(self):
        """Установка соединения с базой данных"""
        try:
            # Проверяем наличие переменных окружения
            required_vars = ['DATABASE_HOST', 'DATABASE_NAME', 'DATABASE_USER', 'DATABASE_PASSWORD']
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            
            if missing_vars:
                logger.warning(f"Отсутствуют переменные окружения: {missing_vars}")
                logger.warning("База данных недоступна, работаем в режиме совместимости")
                return None
            
            # Получаем параметры подключения из переменных окружения
            self.connection = psycopg2.connect(
                host=os.getenv('DATABASE_HOST'),
                port=os.getenv('DATABASE_PORT', '5432'),
                database=os.getenv('DATABASE_NAME'),
                user=os.getenv('DATABASE_USER'),
                password=os.getenv('DATABASE_PASSWORD'),
                cursor_factory=RealDictCursor
            )
            logger.info("Успешное подключение к базе данных")
        except Exception as e:
            logger.error(f"Ошибка подключения к базе данных: {e}")
            logger.warning("Продолжаем работу без базы данных")
            return None
    
    def get_connection(self):
        """Получение соединения с базой данных"""
        if self.connection is None or self.connection.closed:
            self.connect()
        return self.connection
    
    def is_available(self):
        """Проверка доступности базы данных"""
        return self.get_connection() is not None
    
    def execute_query(self, query: str, params: tuple = None, fetch: bool = False) -> Optional[List[Dict]]:
        """Выполнение SQL запроса"""
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                if fetch:
                    return cursor.fetchall()
                conn.commit()
                return cursor.rowcount
        except Exception as e:
            logger.error(f"Ошибка выполнения запроса: {e}")
            conn.rollback()
            raise
    
    def close(self):
        """Закрытие соединения с базой данных"""
        if self.connection:
            self.connection.close()

# Глобальный экземпляр менеджера базы данных
db_manager = DatabaseManager()

# ============ ФУНКЦИИ ДЛЯ РАБОТЫ С ПОЛЬЗОВАТЕЛЯМИ ============

def get_user_by_telegram_id(telegram_id: int) -> Optional[Dict]:
    """Получение пользователя по Telegram ID"""
    if not db_manager.is_available():
        return None
    
    query = """
        SELECT * FROM users 
        WHERE telegram_id = %s AND is_active = TRUE
    """
    result = db_manager.execute_query(query, (telegram_id,), fetch=True)
    return result[0] if result else None

def create_user(telegram_id: int, username: str = None, folder_name: str = None, role: str = "user") -> bool:
    """Создание нового пользователя"""
    if not db_manager.is_available():
        return False
    
    try:
        query = """
            INSERT INTO users (telegram_id, username, folder_name, role)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (telegram_id) DO UPDATE SET
                username = EXCLUDED.username,
                folder_name = EXCLUDED.folder_name,
                role = EXCLUDED.role,
                is_active = TRUE,
                updated_at = CURRENT_TIMESTAMP
        """
        db_manager.execute_query(query, (telegram_id, username, folder_name, role))
        
        # Создаем стандартные категории для нового пользователя
        user = get_user_by_telegram_id(telegram_id)
        if user:
            create_default_categories(user['id'])
        
        return True
    except Exception as e:
        logger.error(f"Ошибка создания пользователя: {e}")
        return False

def get_all_users() -> List[Dict]:
    """Получение всех активных пользователей"""
    query = "SELECT * FROM users WHERE is_active = TRUE ORDER BY created_at"
    return db_manager.execute_query(query, fetch=True) or []

# ============ ФУНКЦИИ ДЛЯ РАБОТЫ С КАТЕГОРИЯМИ ============

def create_default_categories(user_id: int) -> bool:
    """Создание стандартных категорий для пользователя"""
    default_categories = [
        ("Продукты", "expense", "#e74c3c", "🛒"),
        ("Транспорт", "expense", "#3498db", "🚗"),
        ("Развлечения", "expense", "#f39c12", "🎮"),
        ("Здоровье", "expense", "#e91e63", "🏥"),
        ("Одежда", "expense", "#9b59b6", "👕"),
        ("Дом", "expense", "#1abc9c", "🏠"),
        ("Образование", "expense", "#34495e", "📚"),
        ("Другое", "expense", "#95a5a6", "📦"),
        ("Зарплата", "income", "#27ae60", "💰"),
        ("Подработка", "income", "#2ecc71", "💼")
    ]
    
    try:
        query = """
            INSERT INTO user_categories (user_id, category_name, category_type, color, icon)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (user_id, category_name) DO NOTHING
        """
        for name, cat_type, color, icon in default_categories:
            db_manager.execute_query(query, (user_id, name, cat_type, color, icon))
        return True
    except Exception as e:
        logger.error(f"Ошибка создания категорий: {e}")
        return False

def get_user_categories(user_id: int, category_type: str = "expense") -> List[Dict]:
    """Получение категорий пользователя"""
    query = """
        SELECT * FROM user_categories 
        WHERE user_id = %s AND category_type = %s
        ORDER BY category_name
    """
    return db_manager.execute_query(query, (user_id, category_type), fetch=True) or []

def add_user_category(user_id: int, category_name: str, category_type: str = "expense", 
                     color: str = "#3498db", icon: str = "📦") -> bool:
    """Добавление новой категории пользователю"""
    try:
        query = """
            INSERT INTO user_categories (user_id, category_name, category_type, color, icon)
            VALUES (%s, %s, %s, %s, %s)
        """
        db_manager.execute_query(query, (user_id, category_name, category_type, color, icon))
        return True
    except Exception as e:
        logger.error(f"Ошибка добавления категории: {e}")
        return False

# ============ ФУНКЦИИ ДЛЯ РАБОТЫ С РАСХОДАМИ ============

def add_expense(user_id: int, category_id: int, amount: float, description: str = None, expense_date: date = None) -> bool:
    """Добавление расхода"""
    if expense_date is None:
        expense_date = date.today()
    
    try:
        query = """
            INSERT INTO expenses (user_id, category_id, amount, description, date)
            VALUES (%s, %s, %s, %s, %s)
        """
        db_manager.execute_query(query, (user_id, category_id, amount, description, expense_date))
        return True
    except Exception as e:
        logger.error(f"Ошибка добавления расхода: {e}")
        return False

def get_user_expenses(user_id: int, start_date: date = None, end_date: date = None) -> List[Dict]:
    """Получение расходов пользователя за период"""
    query = """
        SELECT e.*, uc.category_name, uc.color, uc.icon
        FROM expenses e
        LEFT JOIN user_categories uc ON e.category_id = uc.id
        WHERE e.user_id = %s
    """
    params = [user_id]
    
    if start_date:
        query += " AND e.date >= %s"
        params.append(start_date)
    
    if end_date:
        query += " AND e.date <= %s"
        params.append(end_date)
    
    query += " ORDER BY e.date DESC"
    
    return db_manager.execute_query(query, tuple(params), fetch=True) or []

def get_expenses_by_category(user_id: int, start_date: date = None, end_date: date = None) -> List[Dict]:
    """Получение расходов по категориям за период"""
    query = """
        SELECT 
            uc.category_name,
            uc.color,
            uc.icon,
            SUM(e.amount) as total_amount,
            COUNT(e.id) as expense_count
        FROM expenses e
        LEFT JOIN user_categories uc ON e.category_id = uc.id
        WHERE e.user_id = %s
    """
    params = [user_id]
    
    if start_date:
        query += " AND e.date >= %s"
        params.append(start_date)
    
    if end_date:
        query += " AND e.date <= %s"
        params.append(end_date)
    
    query += " GROUP BY uc.id, uc.category_name, uc.color, uc.icon ORDER BY total_amount DESC"
    
    return db_manager.execute_query(query, tuple(params), fetch=True) or []

# ============ ФУНКЦИИ ДЛЯ РАБОТЫ С ПЛАНАМИ БЮДЖЕТА ============

def create_budget_plan(user_id: int, plan_name: str, total_amount: float, 
                      start_date: date, end_date: date, categories: Dict = None) -> bool:
    """Создание плана бюджета"""
    try:
        query = """
            INSERT INTO budget_plans (user_id, plan_name, total_amount, start_date, end_date, categories)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        categories_json = json.dumps(categories) if categories else None
        db_manager.execute_query(query, (user_id, plan_name, total_amount, start_date, end_date, categories_json))
        return True
    except Exception as e:
        logger.error(f"Ошибка создания плана бюджета: {e}")
        return False

def get_user_budget_plans(user_id: int, active_only: bool = True) -> List[Dict]:
    """Получение планов бюджета пользователя"""
    query = """
        SELECT * FROM budget_plans 
        WHERE user_id = %s
    """
    params = [user_id]
    
    if active_only:
        query += " AND is_active = TRUE"
    
    query += " ORDER BY created_at DESC"
    
    return db_manager.execute_query(query, tuple(params), fetch=True) or []

def update_budget_plan_spent(user_id: int, plan_id: int, spent_amount: float) -> bool:
    """Обновление потраченной суммы в плане бюджета"""
    try:
        query = """
            UPDATE budget_plans 
            SET spent_amount = %s, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s AND user_id = %s
        """
        db_manager.execute_query(query, (spent_amount, plan_id, user_id))
        return True
    except Exception as e:
        logger.error(f"Ошибка обновления плана бюджета: {e}")
        return False

# ============ ФУНКЦИИ ДЛЯ РАБОТЫ С НАПОМИНАНИЯМИ ============

def add_reminder(user_id: int, title: str, description: str = None, 
                reminder_date: date = None, reminder_time: time = None,
                is_recurring: bool = False, recurring_pattern: str = None) -> bool:
    """Добавление напоминания"""
    if reminder_date is None:
        reminder_date = date.today()
    
    try:
        query = """
            INSERT INTO reminders (user_id, title, description, reminder_date, reminder_time, is_recurring, recurring_pattern)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        db_manager.execute_query(query, (user_id, title, description, reminder_date, reminder_time, is_recurring, recurring_pattern))
        return True
    except Exception as e:
        logger.error(f"Ошибка добавления напоминания: {e}")
        return False

def get_user_reminders(user_id: int, upcoming_only: bool = True) -> List[Dict]:
    """Получение напоминаний пользователя"""
    query = """
        SELECT * FROM reminders 
        WHERE user_id = %s
    """
    params = [user_id]
    
    if upcoming_only:
        query += " AND reminder_date >= CURRENT_DATE AND is_completed = FALSE"
    
    query += " ORDER BY reminder_date, reminder_time"
    
    return db_manager.execute_query(query, tuple(params), fetch=True) or []

# ============ ФУНКЦИИ ДЛЯ РАБОТЫ С НАСТРОЙКАМИ ============

def get_user_setting(user_id: int, setting_key: str) -> Optional[str]:
    """Получение настройки пользователя"""
    query = """
        SELECT setting_value FROM user_settings 
        WHERE user_id = %s AND setting_key = %s
    """
    result = db_manager.execute_query(query, (user_id, setting_key), fetch=True)
    return result[0]['setting_value'] if result else None

def set_user_setting(user_id: int, setting_key: str, setting_value: str) -> bool:
    """Установка настройки пользователя"""
    try:
        query = """
            INSERT INTO user_settings (user_id, setting_key, setting_value)
            VALUES (%s, %s, %s)
            ON CONFLICT (user_id, setting_key) 
            DO UPDATE SET setting_value = EXCLUDED.setting_value, updated_at = CURRENT_TIMESTAMP
        """
        db_manager.execute_query(query, (user_id, setting_key, setting_value))
        return True
    except Exception as e:
        logger.error(f"Ошибка установки настройки: {e}")
        return False

# ============ ФУНКЦИИ МИГРАЦИИ ============

def migrate_user_data_from_files(user_folder_path: str, telegram_id: int) -> bool:
    """Миграция данных пользователя из файловой системы в базу данных"""
    try:
        # Создаем пользователя в БД
        user = get_user_by_telegram_id(telegram_id)
        if not user:
            # Извлекаем username из пути папки
            folder_name = os.path.basename(user_folder_path)
            username = folder_name.split('_')[0] if '_' in folder_name else folder_name
            create_user(telegram_id, username, folder_name)
            user = get_user_by_telegram_id(telegram_id)
        
        if not user:
            return False
        
        user_id = user['id']
        
        # Мигрируем категории
        categories_file = os.path.join(user_folder_path, "user_categories.json")
        if os.path.exists(categories_file):
            with open(categories_file, 'r', encoding='utf-8') as f:
                categories_data = json.load(f)
                for category in categories_data:
                    add_user_category(
                        user_id, 
                        category['name'], 
                        category.get('type', 'expense'),
                        category.get('color', '#3498db'),
                        category.get('icon', '📦')
                    )
        
        # Мигрируем расходы
        expenses_file = os.path.join(user_folder_path, "data", "expenses.json")
        if os.path.exists(expenses_file):
            with open(expenses_file, 'r', encoding='utf-8') as f:
                expenses_data = json.load(f)
                for expense in expenses_data:
                    # Находим категорию по имени
                    category = get_user_categories(user_id)
                    category_id = None
                    for cat in category:
                        if cat['category_name'] == expense.get('category'):
                            category_id = cat['id']
                            break
                    
                    if category_id:
                        expense_date = datetime.strptime(expense['date'], '%Y-%m-%d').date()
                        add_expense(
                            user_id,
                            category_id,
                            expense['amount'],
                            expense.get('description'),
                            expense_date
                        )
        
        # Мигрируем планы бюджета
        budget_plans_file = os.path.join(user_folder_path, "budget_plans.json")
        if os.path.exists(budget_plans_file):
            with open(budget_plans_file, 'r', encoding='utf-8') as f:
                plans_data = json.load(f)
                for plan in plans_data:
                    start_date = datetime.strptime(plan['start_date'], '%Y-%m-%d').date()
                    end_date = datetime.strptime(plan['end_date'], '%Y-%m-%d').date()
                    create_budget_plan(
                        user_id,
                        plan['name'],
                        plan['total_amount'],
                        start_date,
                        end_date,
                        plan.get('categories')
                    )
        
        # Мигрируем напоминания
        reminders_file = os.path.join(user_folder_path, "reminders.json")
        if os.path.exists(reminders_file):
            with open(reminders_file, 'r', encoding='utf-8') as f:
                reminders_data = json.load(f)
                for reminder in reminders_data:
                    reminder_date = datetime.strptime(reminder['date'], '%Y-%m-%d').date()
                    reminder_time = datetime.strptime(reminder.get('time', '00:00'), '%H:%M').time()
                    add_reminder(
                        user_id,
                        reminder['title'],
                        reminder.get('description'),
                        reminder_date,
                        reminder_time,
                        reminder.get('recurring', False),
                        reminder.get('pattern')
                    )
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка миграции данных пользователя: {e}")
        return False
