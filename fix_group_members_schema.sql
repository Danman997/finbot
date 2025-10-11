-- Исправление схемы таблицы group_members для поддержки больших Telegram ID
-- PostgreSQL

-- Проверяем и исправляем типы данных
DO $$
BEGIN
    -- Изменяем тип user_id на BIGINT, если он не BIGINT
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'group_members' 
        AND column_name = 'user_id' 
        AND data_type != 'bigint'
    ) THEN
        ALTER TABLE group_members ALTER COLUMN user_id TYPE BIGINT;
        RAISE NOTICE 'Изменен тип user_id на BIGINT';
    END IF;
    
    -- Изменяем тип group_id на INTEGER, если он не INTEGER
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'group_members' 
        AND column_name = 'group_id' 
        AND data_type != 'integer'
    ) THEN
        ALTER TABLE group_members ALTER COLUMN group_id TYPE INTEGER;
        RAISE NOTICE 'Изменен тип group_id на INTEGER';
    END IF;
    
    -- Убеждаемся, что внешние ключи правильно настроены
    -- Удаляем существующие ограничения, если они есть
    IF EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE table_name = 'group_members' 
        AND constraint_name LIKE '%user_id%'
    ) THEN
        ALTER TABLE group_members DROP CONSTRAINT IF EXISTS group_members_user_id_fkey;
    END IF;
    
    IF EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE table_name = 'group_members' 
        AND constraint_name LIKE '%group_id%'
    ) THEN
        ALTER TABLE group_members DROP CONSTRAINT IF EXISTS group_members_group_id_fkey;
    END IF;
    
    -- Добавляем правильные внешние ключи
    ALTER TABLE group_members 
    ADD CONSTRAINT group_members_user_id_fkey 
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;
    
    ALTER TABLE group_members 
    ADD CONSTRAINT group_members_group_id_fkey 
    FOREIGN KEY (group_id) REFERENCES groups(id) ON DELETE CASCADE;
    
    RAISE NOTICE 'Внешние ключи обновлены';
END $$;

-- Проверяем, что таблица groups тоже имеет правильные типы
DO $$
BEGIN
    -- Изменяем тип admin_user_id на BIGINT в таблице groups
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'groups' 
        AND column_name = 'admin_user_id' 
        AND data_type != 'bigint'
    ) THEN
        ALTER TABLE groups ALTER COLUMN admin_user_id TYPE BIGINT;
        RAISE NOTICE 'Изменен тип admin_user_id на BIGINT в таблице groups';
    END IF;
END $$;

-- Создаем индексы для оптимизации
CREATE INDEX IF NOT EXISTS idx_group_members_user_id ON group_members(user_id);
CREATE INDEX IF NOT EXISTS idx_group_members_group_id ON group_members(group_id);
CREATE INDEX IF NOT EXISTS idx_group_members_user_group ON group_members(user_id, group_id);

-- Проверяем результат
SELECT 
    table_name,
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns 
WHERE table_name IN ('group_members', 'groups', 'users')
AND column_name IN ('user_id', 'group_id', 'admin_user_id', 'id')
ORDER BY table_name, column_name;
