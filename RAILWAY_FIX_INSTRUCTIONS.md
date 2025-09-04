# 🔧 Исправление проблемы с таблицами в Railway

## Проблема
При попытке использовать функции админ-панели (статистика, управление папками) возникает ошибка:
```
relation "user_folders" does not exist
```

## Причина
В базе данных Railway отсутствуют новые таблицы для системы управления пользователями.

## Решение

### Вариант 1: Автоматическое исправление (рекомендуется)
1. **Перезапустите бота на Railway** - таблицы создадутся автоматически при запуске
2. Или выполните команду в Railway:
   ```bash
   python test_db_setup.py
   ```

### Вариант 2: Ручное создание таблиц
Подключитесь к базе данных Railway и выполните SQL:

```sql
-- Создание таблиц для системы управления пользователями
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
);

CREATE TABLE IF NOT EXISTS user_categories (
    id SERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    category_name VARCHAR(100) NOT NULL,
    keywords TEXT[] DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, category_name)
);

CREATE TABLE IF NOT EXISTS user_settings (
    id SERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    setting_key VARCHAR(100) NOT NULL,
    setting_value JSONB,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, setting_key)
);

CREATE TABLE IF NOT EXISTS user_logs (
    id SERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    log_level VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_data (
    id SERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    data_type VARCHAR(50) NOT NULL,
    data_content JSONB NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, data_type)
);

CREATE TABLE IF NOT EXISTS user_backups (
    id SERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    backup_name VARCHAR(100) NOT NULL,
    backup_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

## Что изменилось

### ✅ Обновлена функция `init_db()`
- Добавлено создание всех необходимых таблиц при запуске бота
- Таблицы создаются автоматически при первом запуске

### ✅ Исправлены функции создания папок
- `create_user_folder()` теперь работает с базой данных
- `save_user_data()` и `load_user_data()` используют PostgreSQL
- `create_user_backup()` сохраняет резервные копии в БД

### ✅ Обновлена админ-панель
- Статистика системы работает с базой данных
- Управление папками показывает данные из БД
- Все функции адаптированы для облачной среды

## Проверка исправления

После применения исправления:

1. **Перезапустите бота на Railway**
2. **Попробуйте добавить пользователя через админ-панель:**
   - Войдите как администратор
   - Нажмите "👥 Добавить пользователя"
   - Введите имя пользователя
   - Введите название папки
   - Выберите роль

3. **Проверьте функции админ-панели:**
   - "📊 Статистика системы" - должна показать статистику
   - "📁 Управление папками" - должна показать созданные папки

## Результат

После исправления:
- ✅ Все таблицы созданы в базе данных Railway
- ✅ Пользователи могут создавать персональные папки
- ✅ Данные сохраняются в облаке 24/7
- ✅ Админ-панель работает корректно
- ✅ Система полностью адаптирована для Railway
