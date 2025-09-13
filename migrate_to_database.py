"""
Скрипт миграции данных из файловой системы в базу данных PostgreSQL
Выполняет переход от архитектуры с отдельными папками к единой БД
"""

import os
import json
import sys
from datetime import datetime
from database import db_manager, migrate_user_data_from_files, get_user_by_telegram_id
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_authorized_users():
    """Загрузка списка авторизованных пользователей"""
    try:
        with open('authorized_users.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("Файл authorized_users.json не найден")
        return {}
    except Exception as e:
        logger.error(f"Ошибка загрузки авторизованных пользователей: {e}")
        return {}

def get_user_folder_path(user_id: int):
    """Получение пути к папке пользователя"""
    users_dir = "users"
    if not os.path.exists(users_dir):
        return None
    
    # Ищем папку пользователя по ID
    for folder in os.listdir(users_dir):
        folder_path = os.path.join(users_dir, folder)
        if os.path.isdir(folder_path):
            # Проверяем, содержит ли имя папки ID пользователя
            if str(user_id) in folder:
                return folder_path
    
    return None

def migrate_all_users():
    """Миграция всех пользователей из файловой системы в базу данных"""
    logger.info("Начинаем миграцию пользователей в базу данных...")
    
    # Загружаем список авторизованных пользователей
    authorized_users = load_authorized_users()
    if not authorized_users:
        logger.warning("Список авторизованных пользователей пуст")
        return
    
    migrated_count = 0
    failed_count = 0
    
    for user_id, user_data in authorized_users.items():
        try:
            user_id_int = int(user_id)
            logger.info(f"Мигрируем пользователя {user_id} ({user_data.get('username', 'Unknown')})")
            
            # Проверяем, есть ли уже пользователь в БД
            existing_user = get_user_by_telegram_id(user_id_int)
            if existing_user:
                logger.info(f"Пользователь {user_id} уже существует в БД, пропускаем")
                continue
            
            # Получаем путь к папке пользователя
            user_folder_path = get_user_folder_path(user_id_int)
            if not user_folder_path:
                logger.warning(f"Папка пользователя {user_id} не найдена")
                failed_count += 1
                continue
            
            # Мигрируем данные пользователя
            if migrate_user_data_from_files(user_folder_path, user_id_int):
                logger.info(f"Пользователь {user_id} успешно мигрирован")
                migrated_count += 1
            else:
                logger.error(f"Ошибка миграции пользователя {user_id}")
                failed_count += 1
                
        except Exception as e:
            logger.error(f"Ошибка при миграции пользователя {user_id}: {e}")
            failed_count += 1
    
    logger.info(f"Миграция завершена. Успешно: {migrated_count}, Ошибок: {failed_count}")

def create_backup_of_user_folders():
    """Создание резервной копии папок пользователей перед миграцией"""
    logger.info("Создаем резервную копию папок пользователей...")
    
    backup_dir = f"users_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if os.path.exists("users"):
        import shutil
        shutil.copytree("users", backup_dir)
        logger.info(f"Резервная копия создана в папке: {backup_dir}")
        return backup_dir
    else:
        logger.info("Папка users не найдена, резервная копия не нужна")
        return None

def verify_migration():
    """Проверка успешности миграции"""
    logger.info("Проверяем результаты миграции...")
    
    authorized_users = load_authorized_users()
    if not authorized_users:
        logger.warning("Нет пользователей для проверки")
        return
    
    verified_count = 0
    total_count = len(authorized_users)
    
    for user_id in authorized_users.keys():
        try:
            user_id_int = int(user_id)
            user = get_user_by_telegram_id(user_id_int)
            if user:
                logger.info(f"✓ Пользователь {user_id} найден в БД")
                verified_count += 1
            else:
                logger.error(f"✗ Пользователь {user_id} НЕ найден в БД")
        except Exception as e:
            logger.error(f"Ошибка проверки пользователя {user_id}: {e}")
    
    logger.info(f"Проверка завершена. Найдено в БД: {verified_count}/{total_count}")

def main():
    """Основная функция миграции"""
    print("=" * 60)
    print("МИГРАЦИЯ ФИНАНСОВОГО БОТА В БАЗУ ДАННЫХ")
    print("=" * 60)
    
    # Проверяем подключение к БД
    try:
        db_manager.get_connection()
        logger.info("✓ Подключение к базе данных успешно")
    except Exception as e:
        logger.error(f"✗ Ошибка подключения к базе данных: {e}")
        print("\nУбедитесь, что:")
        print("1. PostgreSQL запущен")
        print("2. База данных создана")
        print("3. Переменные окружения настроены")
        print("4. Схема базы данных создана (database_schema.sql)")
        return
    
    # Создаем резервную копию
    backup_dir = create_backup_of_user_folders()
    
    # Запрашиваем подтверждение
    print(f"\nВНИМАНИЕ! Будет выполнена миграция всех пользователей в базу данных.")
    print(f"Резервная копия создана в папке: {backup_dir}")
    
    response = input("\nПродолжить миграцию? (yes/no): ").lower().strip()
    if response != 'yes':
        print("Миграция отменена пользователем")
        return
    
    # Выполняем миграцию
    try:
        migrate_all_users()
        verify_migration()
        
        print("\n" + "=" * 60)
        print("МИГРАЦИЯ ЗАВЕРШЕНА!")
        print("=" * 60)
        print("Теперь можно:")
        print("1. Протестировать работу бота с новой БД")
        print("2. При необходимости удалить старые папки пользователей")
        print("3. Обновить код бота для работы с БД")
        
    except Exception as e:
        logger.error(f"Критическая ошибка миграции: {e}")
        print(f"\nОшибка миграции: {e}")
        print("Проверьте логи и при необходимости восстановите из резервной копии")

if __name__ == "__main__":
    main()
