#!/usr/bin/env python3
"""
Скрипт для очистки таблиц с 'mama' и настройки доступа к основным таблицам
для пользователей 651498165 и 498410375
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_db_connection():
    """Получает подключение к базе данных"""
    try:
        # Читаем переменные окружения
        host = os.getenv('DATABASE_HOST', os.getenv('PGHOST', 'localhost'))
        port = os.getenv('DATABASE_PORT', os.getenv('PGPORT', '5432'))
        database = os.getenv('DATABASE_NAME', os.getenv('PGDATABASE', 'postgres'))
        user = os.getenv('DATABASE_USER', os.getenv('PGUSER', 'postgres'))
        password = os.getenv('DATABASE_PASSWORD', os.getenv('PGPASSWORD', ''))
        
        # Формируем строку подключения
        database_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        
        conn = psycopg2.connect(database_url)
        logger.info("Успешное подключение к базе данных")
        return conn
    except Exception as e:
        logger.error(f"Ошибка подключения к базе данных: {e}")
        return None

def get_tables_with_mama():
    """Получает список всех таблиц с 'mama' в названии"""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name LIKE '%mama%'
        """)
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tables
    except Exception as e:
        logger.error(f"Ошибка получения списка таблиц: {e}")
        return []

def drop_mama_tables():
    """Удаляет все таблицы с 'mama' в названии"""
    tables = get_tables_with_mama()
    if not tables:
        logger.info("Таблицы с 'mama' не найдены")
        return True
    
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        for table in tables:
            logger.info(f"Удаляем таблицу: {table}")
            cursor.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        
        conn.commit()
        conn.close()
        logger.info(f"Успешно удалено {len(tables)} таблиц с 'mama'")
        return True
    except Exception as e:
        logger.error(f"Ошибка удаления таблиц: {e}")
        return False

def ensure_main_tables_exist():
    """Создает основные таблицы если их нет"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        # Создаем таблицу users
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
        
        # Создаем таблицу user_categories
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
        
        # Создаем таблицу expenses
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
        
        # Создаем таблицу budget_plans
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
        
        # Создаем таблицу reminders
        cursor.execute("""
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
        """)
        
        # Создаем таблицу user_settings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_settings (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                setting_key VARCHAR(255) NOT NULL,
                setting_value TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, setting_key)
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("Основные таблицы созданы/проверены")
        return True
    except Exception as e:
        logger.error(f"Ошибка создания основных таблиц: {e}")
        return False

def create_special_users():
    """Создает специальных пользователей с полным доступом"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        # Создаем пользователя 651498165
        cursor.execute("""
            INSERT INTO users (telegram_id, username, folder_name, role, is_active)
            VALUES (651498165, 'special_user_1', 'special_user_1', 'admin', TRUE)
            ON CONFLICT (telegram_id) DO UPDATE SET
                role = 'admin',
                is_active = TRUE,
                updated_at = CURRENT_TIMESTAMP
        """)
        
        # Создаем пользователя 498410375
        cursor.execute("""
            INSERT INTO users (telegram_id, username, folder_name, role, is_active)
            VALUES (498410375, 'special_user_2', 'special_user_2', 'admin', TRUE)
            ON CONFLICT (telegram_id) DO UPDATE SET
                role = 'admin',
                is_active = TRUE,
                updated_at = CURRENT_TIMESTAMP
        """)
        
        conn.commit()
        conn.close()
        logger.info("Специальные пользователи созданы/обновлены")
        return True
    except Exception as e:
        logger.error(f"Ошибка создания специальных пользователей: {e}")
        return False

def main():
    """Основная функция"""
    logger.info("Начинаем очистку таблиц с 'mama' и настройку доступа")
    
    # 1. Удаляем таблицы с 'mama'
    if not drop_mama_tables():
        logger.error("Ошибка удаления таблиц с 'mama'")
        return False
    
    # 2. Создаем основные таблицы
    if not ensure_main_tables_exist():
        logger.error("Ошибка создания основных таблиц")
        return False
    
    # 3. Создаем специальных пользователей
    if not create_special_users():
        logger.error("Ошибка создания специальных пользователей")
        return False
    
    logger.info("Очистка и настройка завершены успешно!")
    return True

if __name__ == "__main__":
    main()
