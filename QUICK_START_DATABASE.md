# Быстрый старт: Миграция в базу данных

## Что изменилось?

Мы переходим от **отдельных папок для каждого пользователя** к **единой базе данных PostgreSQL** с связанными таблицами.

### До (старая архитектура):
```
users/
├── Пользователь1_123456789_Папка1/
│   ├── data/expenses.json
│   ├── budget_plans.json
│   ├── user_categories.json
│   └── ...
└── Пользователь2_987654321_Папка2/
    ├── data/expenses.json
    └── ...
```

### После (новая архитектура):
```
База данных PostgreSQL:
├── users (главная таблица)
├── expenses (расходы)
├── user_categories (категории)
├── budget_plans (планы)
├── reminders (напоминания)
└── user_settings (настройки)
```

## Быстрая настройка (5 минут)

### 1. Установите PostgreSQL
```bash
# Windows: скачайте с https://www.postgresql.org/download/windows/
# Ubuntu: sudo apt install postgresql postgresql-contrib
```

### 2. Создайте базу данных
```sql
-- Откройте psql
sudo -u postgres psql

-- Создайте БД
CREATE DATABASE finbot_db;
CREATE USER finbot_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE finbot_db TO finbot_user;
```

### 3. Создайте таблицы
```bash
psql -d finbot_db -f database_schema.sql
```

### 4. Настройте переменные окружения
Создайте файл `.env`:
```env
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=finbot_db
DATABASE_USER=finbot_user
DATABASE_PASSWORD=your_password
```

### 5. Протестируйте подключение
```bash
python test_database.py
```

### 6. Выполните миграцию
```bash
python migrate_to_database.py
```

## Что дальше?

После успешной миграции нужно обновить `bot.py` для работы с БД вместо файлов.

## Преимущества новой архитектуры

- ✅ **Стандартный подход** - используется везде в индустрии
- ✅ **Лучшая производительность** - индексы, оптимизация запросов
- ✅ **Надежность** - ACID транзакции, целостность данных
- ✅ **Масштабируемость** - легко добавлять новых пользователей
- ✅ **Аналитика** - сложные запросы и отчеты
- ✅ **Резервное копирование** - один файл дампа вместо тысяч JSON

## Поддержка

Если что-то не работает:
1. Проверьте логи
2. Убедитесь в правильности настроек БД
3. Запустите `test_database.py` для диагностики
4. При необходимости восстановите из резервной копии

---

**Готово!** Теперь у вас современная архитектура с базой данных! 🎉
