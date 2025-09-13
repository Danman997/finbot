"""
Тестовый скрипт для проверки работы с базой данных
Проверяет основные функции database.py
"""

import os
from datetime import date, datetime, time
from database import *

def test_database_connection():
    """Тест подключения к базе данных"""
    print("🔍 Тестируем подключение к базе данных...")
    try:
        conn = db_manager.get_connection()
        print("✅ Подключение к базе данных успешно")
        return True
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        return False

def test_user_operations():
    """Тест операций с пользователями"""
    print("\n👤 Тестируем операции с пользователями...")
    
    test_telegram_id = 999999999
    
    try:
        # Создаем тестового пользователя
        print(f"Создаем тестового пользователя {test_telegram_id}...")
        success = create_user(test_telegram_id, "test_user", "TestFolder")
        if success:
            print("✅ Пользователь создан успешно")
        else:
            print("❌ Ошибка создания пользователя")
            return False
        
        # Получаем пользователя
        user = get_user_by_telegram_id(test_telegram_id)
        if user:
            print(f"✅ Пользователь найден: {user['username']} (ID: {user['id']})")
            user_id = user['id']
        else:
            print("❌ Пользователь не найден")
            return False
        
        # Проверяем создание категорий
        categories = get_user_categories(user_id)
        if categories:
            print(f"✅ Созданы категории: {len(categories)} шт.")
            for cat in categories[:3]:  # Показываем первые 3
                print(f"   - {cat['category_name']} ({cat['category_type']})")
        else:
            print("❌ Категории не созданы")
            return False
        
        return user_id
        
    except Exception as e:
        print(f"❌ Ошибка в операциях с пользователями: {e}")
        return False

def test_expense_operations(user_id):
    """Тест операций с расходами"""
    print(f"\n💰 Тестируем операции с расходами для пользователя {user_id}...")
    
    try:
        # Получаем первую категорию
        categories = get_user_categories(user_id, "expense")
        if not categories:
            print("❌ Нет категорий для тестирования")
            return False
        
        category_id = categories[0]['id']
        category_name = categories[0]['category_name']
        
        # Добавляем тестовый расход
        print(f"Добавляем расход в категорию '{category_name}'...")
        success = add_expense(user_id, category_id, 150.75, "Тестовый расход", date.today())
        if success:
            print("✅ Расход добавлен успешно")
        else:
            print("❌ Ошибка добавления расхода")
            return False
        
        # Получаем расходы пользователя
        expenses = get_user_expenses(user_id)
        if expenses:
            print(f"✅ Найдено расходов: {len(expenses)}")
            expense = expenses[0]
            print(f"   - {expense['amount']} руб. в категории '{expense['category_name']}'")
        else:
            print("❌ Расходы не найдены")
            return False
        
        # Тест группировки по категориям
        expenses_by_category = get_expenses_by_category(user_id)
        if expenses_by_category:
            print(f"✅ Группировка по категориям работает: {len(expenses_by_category)} категорий")
            for cat_data in expenses_by_category:
                print(f"   - {cat_data['category_name']}: {cat_data['total_amount']} руб.")
        else:
            print("❌ Группировка по категориям не работает")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка в операциях с расходами: {e}")
        return False

def test_budget_plan_operations(user_id):
    """Тест операций с планами бюджета"""
    print(f"\n📊 Тестируем операции с планами бюджета для пользователя {user_id}...")
    
    try:
        # Создаем тестовый план бюджета
        plan_name = "Тестовый план"
        total_amount = 10000.00
        start_date = date.today()
        end_date = date(2024, 12, 31)
        
        print(f"Создаем план бюджета '{plan_name}'...")
        success = create_budget_plan(user_id, plan_name, total_amount, start_date, end_date)
        if success:
            print("✅ План бюджета создан успешно")
        else:
            print("❌ Ошибка создания плана бюджета")
            return False
        
        # Получаем планы пользователя
        plans = get_user_budget_plans(user_id)
        if plans:
            print(f"✅ Найден план: {plans[0]['plan_name']} ({plans[0]['total_amount']} руб.)")
        else:
            print("❌ Планы бюджета не найдены")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка в операциях с планами бюджета: {e}")
        return False

def test_reminder_operations(user_id):
    """Тест операций с напоминаниями"""
    print(f"\n⏰ Тестируем операции с напоминаниями для пользователя {user_id}...")
    
    try:
        # Добавляем тестовое напоминание
        title = "Тестовое напоминание"
        description = "Это тестовое напоминание"
        reminder_date = date.today()
        reminder_time = time(12, 0)
        
        print(f"Добавляем напоминание '{title}'...")
        success = add_reminder(user_id, title, description, reminder_date, reminder_time)
        if success:
            print("✅ Напоминание добавлено успешно")
        else:
            print("❌ Ошибка добавления напоминания")
            return False
        
        # Получаем напоминания пользователя
        reminders = get_user_reminders(user_id)
        if reminders:
            print(f"✅ Найдено напоминаний: {len(reminders)}")
            reminder = reminders[0]
            print(f"   - {reminder['title']} на {reminder['reminder_date']}")
        else:
            print("❌ Напоминания не найдены")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка в операциях с напоминаниями: {e}")
        return False

def cleanup_test_data(user_id):
    """Очистка тестовых данных"""
    print(f"\n🧹 Очищаем тестовые данные для пользователя {user_id}...")
    
    try:
        # Удаляем пользователя (все связанные данные удалятся автоматически)
        query = "DELETE FROM users WHERE id = %s"
        db_manager.execute_query(query, (user_id,))
        print("✅ Тестовые данные очищены")
        return True
    except Exception as e:
        print(f"❌ Ошибка очистки тестовых данных: {e}")
        return False

def main():
    """Основная функция тестирования"""
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ БАЗЫ ДАННЫХ ФИНАНСОВОГО БОТА")
    print("=" * 60)
    
    # Проверяем переменные окружения
    required_env_vars = ['DATABASE_HOST', 'DATABASE_NAME', 'DATABASE_USER', 'DATABASE_PASSWORD']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Отсутствуют переменные окружения: {', '.join(missing_vars)}")
        print("Создайте файл .env с необходимыми переменными")
        return
    
    # Тест 1: Подключение к БД
    if not test_database_connection():
        return
    
    # Тест 2: Операции с пользователями
    user_id = test_user_operations()
    if not user_id:
        return
    
    # Тест 3: Операции с расходами
    if not test_expense_operations(user_id):
        cleanup_test_data(user_id)
        return
    
    # Тест 4: Операции с планами бюджета
    if not test_budget_plan_operations(user_id):
        cleanup_test_data(user_id)
        return
    
    # Тест 5: Операции с напоминаниями
    if not test_reminder_operations(user_id):
        cleanup_test_data(user_id)
        return
    
    # Очистка тестовых данных
    cleanup_test_data(user_id)
    
    print("\n" + "=" * 60)
    print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
    print("База данных готова к использованию")
    print("=" * 60)

if __name__ == "__main__":
    main()
