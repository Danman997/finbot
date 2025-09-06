-- Исправление структуры таблиц на Railway
-- Выполните этот скрипт в Railway Dashboard -> Postgres -> Data -> Query

-- 1. Исправляем таблицу user_folders
ALTER TABLE user_folders ALTER COLUMN user_id DROP NOT NULL;

-- 2. Исправляем таблицу user_categories  
ALTER TABLE user_categories ALTER COLUMN user_id DROP NOT NULL;

-- 3. Исправляем таблицу user_settings
ALTER TABLE user_settings ALTER COLUMN user_id DROP NOT NULL;

-- 4. Исправляем таблицу user_logs
ALTER TABLE user_logs ALTER COLUMN user_id DROP NOT NULL;

-- 5. Проверяем структуру таблиц
SELECT 
    table_name,
    column_name,
    is_nullable,
    data_type
FROM information_schema.columns 
WHERE table_name IN ('user_folders', 'user_categories', 'user_settings', 'user_logs')
AND column_name = 'user_id'
ORDER BY table_name;

-- 6. Добавляем тестового пользователя
INSERT INTO user_folders (username, user_id, folder_name, role, settings, permissions)
VALUES (
    'Тест5', 
    NULL, 
    'Тест555', 
    'moderator',
    '{"currency": "Tg", "language": "ru", "notifications": true, "auto_classification": true}',
    '{"add_expenses": true, "view_reports": true, "manage_reminders": true, "planning": true, "analytics": true}'
);

-- 7. Проверяем результат
SELECT * FROM user_folders ORDER BY created_at DESC;

