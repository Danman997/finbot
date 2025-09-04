#!/usr/bin/env python3
"""
Скрипт для ручного добавления тестового пользователя
"""

import json
from datetime import datetime

def add_test_user():
    """Добавляет тестового пользователя в локальный файл"""
    try:
        # Загружаем существующих пользователей
        with open("authorized_users.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Добавляем тестового пользователя
        test_user = {
            "username": "Тест2",
            "added_date": datetime.now().isoformat(),
            "status": "active",
            "role": "moderator",
            "folder_name": "Тест234",
            "telegram_id": None
        }
        
        data["users"].append(test_user)
        
        # Сохраняем обновленные данные
        with open("authorized_users.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print("✅ Тестовый пользователь добавлен в локальный файл:")
        print(f"   Username: {test_user['username']}")
        print(f"   Folder: {test_user['folder_name']}")
        print(f"   Role: {test_user['role']}")
        print(f"   Status: {test_user['status']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при добавлении пользователя: {e}")
        return False

if __name__ == "__main__":
    print("👤 Добавление тестового пользователя")
    print("=" * 40)
    
    if add_test_user():
        print("\n✅ Пользователь успешно добавлен!")
        print("📝 Теперь нужно добавить его в базу данных Railway через бота")
    else:
        print("\n❌ Не удалось добавить пользователя")
