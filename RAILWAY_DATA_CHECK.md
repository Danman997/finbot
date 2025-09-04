# 🔍 Проверка данных пользователей на Railway

## 📊 **Где должны быть записи о пользователе "Тест2" с папкой "Тест234":**

### **1. В базе данных PostgreSQL (Railway):**

#### **Таблица `user_folders`:**
```sql
SELECT * FROM user_folders WHERE username = 'Тест2';
```
**Ожидаемый результат:**
- `id`: уникальный ID
- `username`: "Тест2"
- `user_id`: ID пользователя в Telegram (или NULL)
- `folder_name`: "Тест234"
- `role`: "moderator"
- `settings`: JSON с настройками
- `permissions`: JSON с правами доступа
- `created_at`: дата создания

#### **Таблица `user_categories`:**
```sql
SELECT * FROM user_categories WHERE user_id = [ID_пользователя];
```
**Ожидаемый результат:** 8 стандартных категорий для пользователя

#### **Таблица `user_settings`:**
```sql
SELECT * FROM user_settings WHERE user_id = [ID_пользователя];
```
**Ожидаемый результат:** 4 настройки (notifications, currency, language, backup)

#### **Таблица `user_logs`:**
```sql
SELECT * FROM user_logs WHERE user_id = [ID_пользователя];
```
**Ожидаемый результат:** Логи создания пользователя

### **2. В файле `authorized_users.json` (локально):**
```json
{
  "username": "Тест2",
  "added_date": "2025-09-04T21:48:00.000000",
  "status": "active",
  "role": "moderator",
  "folder_name": "Тест234",
  "telegram_id": null
}
```

## ✅ **Текущий статус:**

### **Локально (✅ ГОТОВО):**
- ✅ Пользователь "Тест2" добавлен в `authorized_users.json`
- ✅ Роль: "moderator"
- ✅ Папка: "Тест234"
- ✅ Статус: "active"

### **На Railway (❓ НУЖНО ПРОВЕРИТЬ):**
- ❓ Записи в таблице `user_folders`
- ❓ Категории в таблице `user_categories`
- ❓ Настройки в таблице `user_settings`
- ❓ Логи в таблице `user_logs`

## 🔧 **Как проверить на Railway:**

### **Вариант 1: Через Railway Dashboard**
1. Зайдите в Railway Dashboard
2. Откройте проект "finbot"
3. Перейдите в "Postgres" → "Data"
4. Найдите таблицы `user_folders`, `user_categories`, `user_settings`, `user_logs`
5. Проверьте наличие записей

### **Вариант 2: Через SQL запросы**
Подключитесь к базе данных Railway и выполните:

```sql
-- Проверка всех папок пользователей
SELECT username, folder_name, role, created_at FROM user_folders ORDER BY created_at DESC;

-- Проверка конкретного пользователя
SELECT * FROM user_folders WHERE username = 'Тест2';

-- Проверка категорий
SELECT user_id, category_name FROM user_categories ORDER BY user_id;

-- Проверка настроек
SELECT user_id, setting_key, setting_value FROM user_settings ORDER BY user_id;

-- Проверка логов
SELECT user_id, log_level, message, created_at FROM user_logs ORDER BY created_at DESC LIMIT 10;
```

## 🚨 **Если записи отсутствуют на Railway:**

### **Причина:**
Пользователи были добавлены только локально, но не через бота на Railway.

### **Решение:**
1. **Перезапустите бота на Railway** (чтобы создались таблицы)
2. **Добавьте пользователей через бота:**
   - Войдите как администратор
   - Используйте "👥 Добавить пользователя"
   - Следуйте процессу: имя → папка → роль

### **Альтернативное решение:**
Создайте SQL скрипт для добавления пользователей напрямую в базу данных:

```sql
-- Добавление пользователя Тест2
INSERT INTO user_folders (username, user_id, folder_name, role, settings, permissions)
VALUES (
    'Тест2', 
    0, -- временный ID, будет обновлен при первом входе
    'Тест234', 
    'moderator',
    '{"currency": "Tg", "language": "ru", "notifications": true, "auto_classification": true}',
    '{"add_expenses": true, "view_reports": true, "manage_reminders": true, "planning": true, "analytics": true}'
);
```

## 📋 **Ожидаемый результат:**

После правильного добавления пользователя "Тест2":

1. **В Railway Dashboard** должны появиться записи в таблицах
2. **Функция "📁 Управление папками"** должна показать папку "Тест234"
3. **Функция "📊 Статистика системы"** должна показать 1 папку пользователя
4. **Пользователь сможет войти в бота** и использовать все функции

## 🎯 **Следующие шаги:**

1. Проверьте данные в Railway Dashboard
2. Если записи отсутствуют - перезапустите бота
3. Добавьте пользователей через админ-панель бота
4. Проверьте работу функций админ-панели
