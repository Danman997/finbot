#!/usr/bin/env python3
"""
Скрипт для тестирования создания пользователей
"""

import json
from datetime import datetime

def test_user_creation():
    """Тестирует создание пользователя локально"""
    try:
        # Загружаем существующих пользователей
        with open("authorized_users.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Добавляем тестового пользователя
        test_user = {
            "username": "Тест3",
            "added_date": datetime.now().isoformat(),
            "status": "active",
            "role": "user",
            "folder_name": "Тест345",
            "telegram_id": None
        }
        
        data["users"].append(test_user)
        
        # Сохраняем обновленные данные
        with open("authorized_users.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print("✅ Тестовый пользователь добавлен локально:")
        print(f"   Username: {test_user['username']}")
        print(f"   Folder: {test_user['folder_name']}")
        print(f"   Role: {test_user['role']}")
        print(f"   Status: {test_user['status']}")
        print(f"   Telegram ID: {test_user['telegram_id']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при добавлении пользователя: {e}")
        return False

def show_current_users():
    """Показывает текущих пользователей"""
    try:
        with open("authorized_users.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        print("\n📋 Текущие пользователи:")
        print(f"   Admin ID: {data.get('admin')}")
        print(f"   Users count: {len(data.get('users', []))}")
        
        for i, user in enumerate(data.get("users", []), 1):
            print(f"   {i}. {user.get('username')} ({user.get('role', 'user')})")
            print(f"      Folder: {user.get('folder_name', 'Не задана')}")
            print(f"      Status: {user.get('status', 'unknown')}")
            print(f"      Telegram ID: {user.get('telegram_id', 'Не привязан')}")
            print(f"      Added: {user.get('added_date', 'unknown')}")
            print()
            
    except Exception as e:
        print(f"❌ Ошибка при чтении пользователей: {e}")

if __name__ == "__main__":
    print("🧪 Тестирование создания пользователей")
    print("=" * 50)
    
    show_current_users()
    
    print("\n" + "=" * 50)
    print("➕ Добавление тестового пользователя...")
    
    if test_user_creation():
        print("\n✅ Пользователь успешно добавлен!")
        print("\n📋 Обновленный список пользователей:")
        show_current_users()
        
        print("\n🎯 Следующие шаги:")
        print("1. Перезапустите бота на Railway")
        print("2. Попробуйте добавить пользователя через админ-панель")
        print("3. Проверьте создание записей в базе данных")
    else:
        print("\n❌ Не удалось добавить пользователя")
