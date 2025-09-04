#!/usr/bin/env python3
"""
Скрипт для тестирования создания таблиц в базе данных Railway
"""

import os
import psycopg2
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

def test_db_connection():
    """Тестирует подключение к базе данных"""
    try:
        DATABASE_URL = os.environ.get('DATABASE_URL')
        if not DATABASE_URL:
            print("❌ DATABASE_URL не найден в переменных окружения")
            return False
        
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        # Проверяем существующие таблицы
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        
        tables = cursor.fetchall()
        print("📊 Существующие таблицы в базе данных:")
        for table in tables:
            print(f"   - {table[0]}")
        
        # Проверяем таблицы для системы пользователей
        required_tables = [
            'user_folders', 'user_categories', 'user_settings', 
            'user_logs', 'user_data', 'user_backups'
        ]
        
        existing_table_names = [table[0] for table in tables]
        missing_tables = [table for table in required_tables if table not in existing_table_names]
        
        if missing_tables:
            print(f"\n❌ Отсутствующие таблицы: {missing_tables}")
            return False
        else:
            print(f"\n✅ Все необходимые таблицы существуют!")
            return True
            
        conn.close()
        
    except Exception as e:
        print(f"❌ Ошибка подключения к базе данных: {e}")
        return False

def create_missing_tables():
    """Создает недостающие таблицы"""
    try:
        DATABASE_URL = os.environ.get('DATABASE_URL')
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        # Создаем таблицы для системы управления пользователями
        tables_sql = [
            '''
            CREATE TABLE IF NOT EXISTS user_folders (
                id SERIAL PRIMARY KEY,
                username VARCHAR(100) NOT NULL,
                user_id BIGINT NOT NULL,
                folder_name VARCHAR(100) NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                role VARCHAR(20) DEFAULT 'user',
                settings JSONB DEFAULT '{}',
                permissions JSONB DEFAULT '{}',
                UNIQUE(username, user_id)
            )
            ''',
            '''
            CREATE TABLE IF NOT EXISTS user_categories (
                id SERIAL PRIMARY KEY,
                user_id BIGINT NOT NULL,
                category_name VARCHAR(100) NOT NULL,
                keywords TEXT[] DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, category_name)
            )
            ''',
            '''
            CREATE TABLE IF NOT EXISTS user_settings (
                id SERIAL PRIMARY KEY,
                user_id BIGINT NOT NULL,
                setting_key VARCHAR(100) NOT NULL,
                setting_value JSONB,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, setting_key)
            )
            ''',
            '''
            CREATE TABLE IF NOT EXISTS user_logs (
                id SERIAL PRIMARY KEY,
                user_id BIGINT NOT NULL,
                log_level VARCHAR(20) NOT NULL,
                message TEXT NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
            ''',
            '''
            CREATE TABLE IF NOT EXISTS user_data (
                id SERIAL PRIMARY KEY,
                user_id BIGINT NOT NULL,
                data_type VARCHAR(50) NOT NULL,
                data_content JSONB NOT NULL,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, data_type)
            )
            ''',
            '''
            CREATE TABLE IF NOT EXISTS user_backups (
                id SERIAL PRIMARY KEY,
                user_id BIGINT NOT NULL,
                backup_name VARCHAR(100) NOT NULL,
                backup_data JSONB NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
            '''
        ]
        
        for i, sql in enumerate(tables_sql, 1):
            cursor.execute(sql)
            print(f"✅ Таблица {i}/6 создана")
        
        conn.commit()
        conn.close()
        
        print("\n🎉 Все таблицы успешно созданы!")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при создании таблиц: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Тестирование базы данных Railway...")
    print("=" * 50)
    
    # Проверяем подключение и существующие таблицы
    if test_db_connection():
        print("\n✅ База данных готова к работе!")
    else:
        print("\n🔨 Создаем недостающие таблицы...")
        if create_missing_tables():
            print("\n✅ Таблицы созданы! Проверяем еще раз...")
            test_db_connection()
        else:
            print("\n❌ Не удалось создать таблицы")
