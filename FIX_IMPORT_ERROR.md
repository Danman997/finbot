# Исправление ошибки импорта config

## Проблема
```
ImportError: cannot import name 'config' from 'config' (/app/config/__init__.py)
```

## ✅ Что было исправлено

### 1. Исправлен импорт в utils/logger.py
**Файл**: `utils/logger.py`, строка 9

**Было**:
```python
from config import config
```

**Стало**:
```python
from config.settings import settings
```

### 2. Обновлены ссылки на config
**Файл**: `utils/logger.py`, строки 42 и 65

**Было**:
```python
logger.setLevel(getattr(logging, config.log_level.upper()))
console_handler.setLevel(getattr(logging, config.log_level.upper()))
```

**Стало**:
```python
logger.setLevel(getattr(logging, settings.logging.level.upper()))
console_handler.setLevel(getattr(logging, settings.logging.level.upper()))
```

## 🚀 Результат

Теперь все импорты работают корректно:
- ✅ `config.settings` импортируется успешно
- ✅ `utils.logger` импортируется успешно
- ✅ `bot.py` импортируется успешно

## 📝 Проверка

Локальная проверка:
```bash
python -c "import bot; print('✅ Все импорты успешны')"
```

## ⚠️ Важно

После коммита и пуша на Railway бот должен запуститься без ошибок импорта.

---

**Статус**: ✅ Исправлено  
**Дата**: 2025-10-05
