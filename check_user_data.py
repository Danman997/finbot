#!/usr/bin/env python3
"""
Скрипт для проверки данных пользователей в базе данных Railway
"""

import os
import psycopg2
import json
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

def check_user_in_db():
    """Проверяет наличие пользователя в базе данных"""
    try:
        DATABASE_URL = os.environ.get('DATABASE_URL')
        if not DATABASE_URL:
            print("❌ DATABASE_URL не найден в переменных окружения")
            return
        
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        print("🔍 Проверяем данные пользователей в базе данных...")
        print("=" * 60)
        
        # Проверяем таблицу user_folders
        print("\n📁 Таблица user_folders:")
        cursor.execute("SELECT * FROM user_folders ORDER BY created_at DESC")
        folders = cursor.fetchall()
        
        if folders:
            for folder in folders:
                print(f"   ID: {folder[0]}")
                print(f"   Username: {folder[1]}")
                print(f"   User ID: {folder[2]}")
                print(f"   Folder: {folder[3]}")
                print(f"   Role: {folder[4]}")
                print(f"   Created: {folder[5]}")
                print(f"   Settings: {folder[6]}")
                print(f"   Permissions: {folder[7]}")
                print("   " + "-" * 40)
        else:
            print("   ❌ Записи не найдены")
        
        # Проверяем таблицу user_categories
        print("\n🏷️ Таблица user_categories:")
        cursor.execute("SELECT user_id, category_name, keywords FROM user_categories ORDER BY user_id, category_name")
        categories = cursor.fetchall()
        
        if categories:
            current_user = None
            for cat in categories:
                if cat[0] != current_user:
                    current_user = cat[0]
                    print(f"   User ID {current_user}:")
                print(f"     - {cat[1]}: {cat[2]}")
        else:
            print("   ❌ Записи не найдены")
        
        # Проверяем таблицу user_settings
        print("\n⚙️ Таблица user_settings:")
        cursor.execute("SELECT user_id, setting_key, setting_value FROM user_settings ORDER BY user_id")
        settings = cursor.fetchall()
        
        if settings:
            current_user = None
            for setting in settings:
                if setting[0] != current_user:
                    current_user = setting[0]
                    print(f"   User ID {current_user}:")
                print(f"     - {setting[1]}: {setting[2]}")
        else:
            print("   ❌ Записи не найдены")
        
        # Проверяем таблицу user_logs
        print("\n📝 Таблица user_logs:")
        cursor.execute("SELECT user_id, log_level, message, created_at FROM user_logs ORDER BY created_at DESC LIMIT 10")
        logs = cursor.fetchall()
        
        if logs:
            for log in logs:
                print(f"   User {log[0]} [{log[1]}]: {log[2]} ({log[3]})")
        else:
            print("   ❌ Записи не найдены")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Ошибка при проверке базы данных: {e}")

def check_local_file():
    """Проверяет локальный файл authorized_users.json"""
    try:
        print("\n📄 Локальный файл authorized_users.json:")
        if os.path.exists("authorized_users.json"):
            with open("authorized_users.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            
            print(f"   Admin ID: {data.get('admin')}")
            print(f"   Users count: {len(data.get('users', []))}")
            
            for user in data.get("users", []):
                print(f"   - {user.get('username')} ({user.get('role', 'user')})")
                print(f"     Folder: {user.get('folder_name', 'Не задана')}")
                print(f"     Status: {user.get('status', 'unknown')}")
                print(f"     Added: {user.get('added_date', 'unknown')}")
        else:
            print("   ❌ Файл не найден")
            
    except Exception as e:
        print(f"❌ Ошибка при проверке локального файла: {e}")

if __name__ == "__main__":
    print("🔍 Проверка данных пользователей")
    print("=" * 60)
    
    check_user_in_db()
    check_local_file()
    
    print("\n" + "=" * 60)
    print("✅ Проверка завершена")
