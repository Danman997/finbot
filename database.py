"""
Модуль для работы с базой данных PostgreSQL
Содержит все функции для работы с пользователями и их данными
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from datetime import datetime, date, time
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
    
    # Принудительно создаем таблицу users если её нет
    try:
        conn = db_manager.get_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    telegram_id BIGINT UNIQUE NOT NULL,
                    username VARCHAR(255),
                    folder_name VARCHAR(255),
                    role VARCHAR(50) DEFAULT 'user',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)
            conn.commit()
    except Exception as e:
        logger.error(f"Ошибка создания таблицы users: {e}")
    
    query = """
        INSERT INTO users (telegram_id, username, folder_name, role)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (telegram_id) DO UPDATE SET
            username = EXCLUDED.username,
            folder_name = EXCLUDED.folder_name,
            role = EXCLUDED.role,
            updated_at = CURRENT_TIMESTAMP
    """
    
    result = db_manager.execute_query(query, (telegram_id, username, folder_name, role))
    return result is not None

def get_all_users() -> List[Dict[str, Any]]:
    """Получает всех пользователей из базы данных"""
    if not db_manager.is_available():
        logger.warning("База данных недоступна")
        return []
    
    try:
        query = """
            SELECT id, telegram_id, username, folder_name, role, created_at, updated_at, is_active
            FROM users 
            WHERE is_active = TRUE 
            ORDER BY created_at DESC
        """
        result = db_manager.execute_query(query, fetch=True)
        return result if result else []
    except Exception as e:
        logger.error(f"Ошибка получения всех пользователей: {e}")
        return []

def update_user_role(telegram_id: int, new_role: str) -> bool:
    """Обновляет роль пользователя"""
    if not db_manager.is_available():
        logger.warning("База данных недоступна")
        return False
    
    try:
        query = """
            UPDATE users 
            SET role = %s, updated_at = CURRENT_TIMESTAMP
            WHERE telegram_id = %s
        """
        result = db_manager.execute_query(query, (new_role, telegram_id))
        if result and result > 0:
            logger.info(f"Роль пользователя {telegram_id} обновлена на {new_role}")
            return True
        else:
            logger.warning(f"Пользователь {telegram_id} не найден")
            return False
    except Exception as e:
        logger.error(f"Ошибка обновления роли пользователя {telegram_id}: {e}")
        return False

# ============ ФУНКЦИИ ДЛЯ РАБОТЫ С КАТЕГОРИЯМИ ============

def get_user_categories(user_id: int) -> List[Dict[str, Any]]:
    """Получение категорий пользователя"""
    if not db_manager.is_available():
        logger.warning("База данных недоступна")
        return []
    
    # Принудительно создаем таблицу user_categories если её нет
    try:
        conn = db_manager.get_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_categories (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    category_name VARCHAR(255) NOT NULL,
                    category_type VARCHAR(50) DEFAULT 'expense',
                    color VARCHAR(7) DEFAULT '#3498db',
                    icon VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, category_name)
                )
            """)
            conn.commit()
    except Exception as e:
        logger.error(f"Ошибка создания таблицы user_categories: {e}")
    
    query = """
        SELECT * FROM user_categories 
        WHERE user_id = %s 
        ORDER BY category_name
    """
    result = db_manager.execute_query(query, (user_id,), fetch=True)
    return result if result else []

def create_default_categories(user_id: int) -> bool:
    """Создание категорий по умолчанию для пользователя"""
    if not db_manager.is_available():
        logger.warning("База данных недоступна")
        return False
    
    default_categories = [
        ("Продукты", "expense", "#e74c3c", "🛒"),
        ("Транспорт", "expense", "#3498db", "🚗"),
        ("Развлечения", "expense", "#9b59b6", "🎮"),
        ("Здоровье", "expense", "#2ecc71", "🏥"),
        ("Одежда", "expense", "#f39c12", "👕"),
        ("Дом", "expense", "#34495e", "🏠"),
        ("Другое", "expense", "#95a5a6", "📦")
    ]
    
    try:
        for category_name, category_type, color, icon in default_categories:
            query = """
                INSERT INTO user_categories (user_id, category_name, category_type, color, icon)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (user_id, category_name) DO NOTHING
            """
            db_manager.execute_query(query, (user_id, category_name, category_type, color, icon))
        
        logger.info(f"Созданы категории по умолчанию для пользователя {user_id}")
        return True
    except Exception as e:
        logger.error(f"Ошибка создания категорий по умолчанию: {e}")
        return False

# ============ ФУНКЦИИ ДЛЯ РАБОТЫ С РАСХОДАМИ ============

def add_expense(user_id: int, category_id: int, amount: float, description: str, expense_date: date) -> bool:
    """Добавление расхода"""
    if not db_manager.is_available():
        logger.warning("База данных недоступна")
        return False
    
    # Принудительно создаем таблицу expenses если её нет
    try:
        conn = db_manager.get_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS expenses (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    category_id INTEGER REFERENCES user_categories(id) ON DELETE SET NULL,
                    amount DECIMAL(10,2) NOT NULL,
                    description TEXT,
                    date DATE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    except Exception as e:
        logger.error(f"Ошибка создания таблицы expenses: {e}")
    
    query = """
        INSERT INTO expenses (user_id, category_id, amount, description, date)
        VALUES (%s, %s, %s, %s, %s)
    """
    
    result = db_manager.execute_query(query, (user_id, category_id, amount, description, expense_date))
    return result is not None

# ============ ФУНКЦИИ ДЛЯ РАБОТЫ С ПЛАНАМИ БЮДЖЕТА ============

def get_user_budget_plans(user_id: int) -> List[Dict[str, Any]]:
    """Получение планов бюджета пользователя"""
    if not db_manager.is_available():
        logger.warning("База данных недоступна")
        return []
    
    query = """
        SELECT * FROM budget_plans 
        WHERE user_id = %s AND is_active = TRUE
        ORDER BY created_at DESC
    """
    result = db_manager.execute_query(query, (user_id,), fetch=True)
    return result if result else []

def save_user_budget_plan(user_id: int, plan_name: str, total_amount: float, 
                         start_date: date, end_date: date, categories: List[str] = None) -> bool:
    """Сохранение плана бюджета пользователя"""
    if not db_manager.is_available():
        logger.warning("База данных недоступна")
        return False
    
    # Принудительно создаем таблицу budget_plans если её нет
    try:
        conn = db_manager.get_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS budget_plans (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    plan_name VARCHAR(255) NOT NULL,
                    total_amount DECIMAL(10,2) NOT NULL,
                    spent_amount DECIMAL(10,2) DEFAULT 0.00,
                    start_date DATE NOT NULL,
                    end_date DATE NOT NULL,
                    categories JSONB,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    except Exception as e:
        logger.error(f"Ошибка создания таблицы budget_plans: {e}")
    
    query = """
        INSERT INTO budget_plans (user_id, plan_name, total_amount, start_date, end_date, categories)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    
    categories_json = json.dumps(categories) if categories else None
    result = db_manager.execute_query(query, (user_id, plan_name, total_amount, start_date, end_date, categories_json))
    return result is not None

# ============ ФУНКЦИИ ДЛЯ РАБОТЫ С НАПОМИНАНИЯМИ ============

def get_user_reminders(user_id: int) -> List[Dict[str, Any]]:
    """Получение напоминаний пользователя"""
    if not db_manager.is_available():
        logger.warning("База данных недоступна")
        return []
    
    query = """
        SELECT * FROM reminders 
        WHERE user_id = %s AND is_completed = FALSE
        ORDER BY reminder_date, reminder_time
    """
    result = db_manager.execute_query(query, (user_id,), fetch=True)
    return result if result else []

def add_reminder(user_id: int, title: str, description: str, reminder_date: date, 
                reminder_time: time = None, is_recurring: bool = False, pattern: str = None) -> bool:
    """Добавление напоминания"""
    if not db_manager.is_available():
        logger.warning("База данных недоступна")
        return False
    
    query = """
        INSERT INTO reminders (user_id, title, description, reminder_date, reminder_time, is_recurring, recurring_pattern)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    
    result = db_manager.execute_query(query, (user_id, title, description, reminder_date, reminder_time, is_recurring, pattern))
    return result is not None

def delete_reminder(reminder_id: int) -> bool:
    """Удаление напоминания"""
    query = """
        DELETE FROM reminders WHERE id = %s
    """
    
    result = db_manager.execute_query(query, (reminder_id,))
    return result is not None

# ============ ФУНКЦИИ ДЛЯ РАБОТЫ С НАСТРОЙКАМИ ============

def get_user_settings(user_id: int) -> Dict[str, Any]:
    """Получение настроек пользователя"""
    if not db_manager.is_available():
        logger.warning("База данных недоступна")
        return {}
    
    query = """
        SELECT setting_key, setting_value FROM user_settings 
        WHERE user_id = %s
    """
    result = db_manager.execute_query(query, (user_id,), fetch=True)
    
    settings = {}
    if result:
        for row in result:
            settings[row['setting_key']] = row['setting_value']
    
    return settings

def save_user_setting(user_id: int, key: str, value: str) -> bool:
    """Сохранение настройки пользователя"""
    if not db_manager.is_available():
        logger.warning("База данных недоступна")
        return False
    
    query = """
        INSERT INTO user_settings (user_id, setting_key, setting_value)
        VALUES (%s, %s, %s)
        ON CONFLICT (user_id, setting_key) 
        DO UPDATE SET setting_value = EXCLUDED.setting_value, updated_at = CURRENT_TIMESTAMP
    """
    
    result = db_manager.execute_query(query, (user_id, key, value))
    return result is not None

# ============ ФУНКЦИИ ИНИЦИАЛИЗАЦИИ ============

def init_db():
    """Инициализация базы данных"""
    try:
        db_manager.connect()
        logger.info("База данных инициализирована")
        return True
    except Exception as e:
        logger.error(f"Ошибка инициализации базы данных: {e}")
        return False

def ensure_tables_exist():
    """Создание таблиц если их нет"""
    try:
        if db_manager.is_available():
            conn = db_manager.get_connection()
            if conn:
                cursor = conn.cursor()
                
                # Создаем все таблицы
                tables = [
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        telegram_id BIGINT UNIQUE NOT NULL,
                        username VARCHAR(255),
                        folder_name VARCHAR(255),
                        role VARCHAR(50) DEFAULT 'user',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT TRUE
                    )
                    """,
                    """
                    CREATE TABLE IF NOT EXISTS user_categories (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                        category_name VARCHAR(255) NOT NULL,
                        category_type VARCHAR(50) DEFAULT 'expense',
                        color VARCHAR(7) DEFAULT '#3498db',
                        icon VARCHAR(50),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(user_id, category_name)
                    )
                    """,
                    """
                    CREATE TABLE IF NOT EXISTS expenses (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                        category_id INTEGER REFERENCES user_categories(id) ON DELETE SET NULL,
                        amount DECIMAL(10,2) NOT NULL,
                        description TEXT,
                        date DATE NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """,
                    """
                    CREATE TABLE IF NOT EXISTS budget_plans (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                        plan_name VARCHAR(255) NOT NULL,
                        total_amount DECIMAL(10,2) NOT NULL,
                        spent_amount DECIMAL(10,2) DEFAULT 0.00,
                        start_date DATE NOT NULL,
                        end_date DATE NOT NULL,
                        categories JSONB,
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """,
                    """
                    CREATE TABLE IF NOT EXISTS reminders (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                        title VARCHAR(255) NOT NULL,
                        description TEXT,
                        reminder_date DATE NOT NULL,
                        reminder_time TIME,
                        is_recurring BOOLEAN DEFAULT FALSE,
                        recurring_pattern VARCHAR(50),
                        is_completed BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """,
                    """
                    CREATE TABLE IF NOT EXISTS user_settings (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                        setting_key VARCHAR(255) NOT NULL,
                        setting_value TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(user_id, setting_key)
                    )
                    """
                ]
                
                for table_sql in tables:
                    cursor.execute(table_sql)
                
                conn.commit()
                logger.info("Таблицы созданы/проверены")
        
        return True
    except Exception as e:
        logger.error(f"Ошибка создания таблиц: {e}")
        return False

def migrate_user_data(user_id: int, user_folder_path: str) -> bool:
    """Миграция данных пользователя из JSON файлов в базу данных"""
    if not db_manager.is_available():
        logger.warning("База данных недоступна")
        return False
    
    try:
        # Мигрируем категории
        categories_file = os.path.join(user_folder_path, "user_categories.json")
        if os.path.exists(categories_file):
            with open(categories_file, 'r', encoding='utf-8') as f:
                categories_data = json.load(f)
                for category in categories_data:
                    query = """
                        INSERT INTO user_categories (user_id, category_name, category_type, color, icon)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (user_id, category_name) DO NOTHING
                    """
                    db_manager.execute_query(query, (
                        user_id, 
                        category['name'], 
                        category.get('type', 'expense'),
                        category.get('color', '#3498db'),
                        category.get('icon', '📦')
                    ))
        
        # Мигрируем расходы
        expenses_file = os.path.join(user_folder_path, "data", "expenses.json")
        if os.path.exists(expenses_file):
            with open(expenses_file, 'r', encoding='utf-8') as f:
                expenses_data = json.load(f)
                for expense in expenses_data:
                    # Находим ID категории
                    category_query = """
                        SELECT id FROM user_categories 
                        WHERE user_id = %s AND category_name = %s
                    """
                    category_result = db_manager.execute_query(category_query, (user_id, expense['category']), fetch=True)
                    category_id = category_result[0]['id'] if category_result else None
                    
                    expense_date = datetime.strptime(expense['date'], '%Y-%m-%d').date()
                    query = """
                        INSERT INTO expenses (user_id, category_id, amount, description, date)
                        VALUES (%s, %s, %s, %s, %s)
                    """
                    db_manager.execute_query(query, (
                        user_id, 
                        category_id, 
                        expense['amount'], 
                        expense.get('description', ''), 
                        expense_date
                    ))
        
        # Мигрируем планы бюджета
        budget_file = os.path.join(user_folder_path, "budget_plans.json")
        if os.path.exists(budget_file):
            with open(budget_file, 'r', encoding='utf-8') as f:
                budget_data = json.load(f)
                for plan in budget_data:
                    start_date = datetime.strptime(plan['start_date'], '%Y-%m-%d').date()
                    end_date = datetime.strptime(plan['end_date'], '%Y-%m-%d').date()
                    categories_json = json.dumps(plan.get('categories', []))
                    
                    query = """
                        INSERT INTO budget_plans (user_id, plan_name, total_amount, start_date, end_date, categories)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """
                    db_manager.execute_query(query, (
                        user_id, 
                        plan['name'], 
                        plan['total_amount'], 
                        start_date, 
                        end_date, 
                        categories_json
                    ))
        
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
