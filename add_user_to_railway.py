#!/usr/bin/env python3
"""
Скрипт для ручного добавления пользователя в базу данных Railway
"""

import os
import psycopg2
import json
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

def add_user_to_railway():
    """Добавляет пользователя в базу данных Railway"""
    try:
        DATABASE_URL = os.environ.get('DATABASE_URL')
        if not DATABASE_URL:
            print("❌ DATABASE_URL не найден в переменных окружения")
            print("💡 Убедитесь, что вы запускаете скрипт на Railway или добавили DATABASE_URL локально")
            return False
        
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        print("🔗 Подключение к базе данных Railway...")
        
        # Проверяем существование таблиц
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('user_folders', 'user_categories', 'user_settings', 'user_logs')
        """)
        tables = [row[0] for row in cursor.fetchall()]
        print(f"📋 Найденные таблицы: {tables}")
        
        if 'user_folders' not in tables:
            print("❌ Таблица user_folders не найдена. Создаем...")
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_folders (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(100) NOT NULL,
                    user_id BIGINT,
                    folder_name VARCHAR(100) NOT NULL,
                    role VARCHAR(50) DEFAULT 'user',
                    settings JSONB,
                    permissions JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(username, user_id)
                )
            ''')
            conn.commit()
            print("✅ Таблица user_folders создана")
        
        # Добавляем тестового пользователя
        username = "Тест2"
        folder_name = "Тест234"
        user_id = None  # Будет обновлен при первом входе
        
        print(f"👤 Добавление пользователя: {username}")
        
        # Вставляем данные пользователя
        cursor.execute('''
            INSERT INTO user_folders (username, user_id, folder_name, role, settings, permissions)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (username, user_id) 
            DO UPDATE SET folder_name = EXCLUDED.folder_name
        ''', (
            username, 
            user_id, 
            folder_name, 
            'moderator',
            json.dumps({
                "currency": "Tg",
                "language": "ru", 
                "notifications": True,
                "auto_classification": True
            }),
            json.dumps({
                "add_expenses": True,
                "view_reports": True,
                "manage_reminders": True,
                "planning": True,
                "analytics": True
            })
        ))
        
        # Проверяем результат
        cursor.execute('SELECT COUNT(*) FROM user_folders WHERE username = %s', (username,))
        count = cursor.fetchone()[0]
        
        if count > 0:
            print(f"✅ Пользователь {username} успешно добавлен в базу данных")
            
            # Показываем все записи
            cursor.execute('SELECT * FROM user_folders ORDER BY created_at DESC')
            records = cursor.fetchall()
            
            print(f"\n📊 Всего записей в user_folders: {len(records)}")
            for record in records:
                print(f"   ID: {record[0]}, Username: {record[1]}, Folder: {record[3]}, Role: {record[5]}")
        else:
            print(f"❌ Не удалось добавить пользователя {username}")
        
        conn.commit()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при добавлении пользователя: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Добавление пользователя в базу данных Railway")
    print("=" * 50)
    
    if add_user_to_railway():
        print("\n✅ Пользователь успешно добавлен!")
        print("🎯 Теперь проверьте таблицу user_folders в Railway Dashboard")
    else:
        print("\n❌ Не удалось добавить пользователя")
        print("💡 Убедитесь, что:")
        print("   1. DATABASE_URL настроен правильно")
        print("   2. Бот развернут на Railway")
        print("   3. База данных доступна")
