import os
import logging
import psycopg2
from psycopg2 import sql
from datetime import datetime, timedelta, timezone, date
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
from telegram.ext import Application, CommandHandler, MessageHandler, ConversationHandler, ContextTypes, filters
import matplotlib.pyplot as plt
import io
import re
import schedule
import time
import pandas as pd
import json
import secrets
import string

# Настройки matplotlib для высокого качества
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# --- Логирование ---
log_directory = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_directory, exist_ok=True)
log_file_path = os.path.join(log_directory, 'finbot.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),  # Логи будут записываться в logs/finbot.log
        logging.StreamHandler()  # Логи также будут выводиться в консоль
    ]
)
logger = logging.getLogger(__name__)

# Загружаем переменные окружения из .env файла
load_dotenv()

# --- Настройки бота ---
BOT_TOKEN = os.environ.get('BOT_TOKEN') 
if not BOT_TOKEN:
    logger.error("Ошибка: Токен бота не найден. Установите переменную окружения BOT_TOKEN.")
    exit()

DATABASE_URL = os.environ.get('DATABASE_URL')
if not DATABASE_URL:
    logger.error("Ошибка: URL базы данных не найден. Установите переменную окружения DATABASE_URL.")
    exit()
    
# --- Классификация расходов: гибридный подход (словарь → фуззи → ML) ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import unicodedata

# Если выше в файле больше не будет TRAINING_DATA – оставим пустой,
# чтобы main() мог вызвать train_model(TRAINING_DATA) без ошибок.
TRAINING_DATA = []

# 1) Расширенный словарь категорий с синонимами/однокоренными
CATEGORIES = {
    "Продукты": [
        "хлеб","батон","булочка","багет","лаваш","пицца","пирог","пирожок","печенье","торт","круассан","бублик","сухарики","пряники","крекер",
        "молоко","кефир","сливки","сметана","йогурт","творог","сыр","масло сливочное","масло подсолнечное","маргарин",
        "яйца","мясо","говядина","свинина","баранина","курица","индейка","утка","рыба","лосось","форель","треска","минтай","тунец","икра",
        "колбаса","сосиски","сардельки","бекон","шашлык","консервы","тушенка","паштет",
        "гречка","рис","перловка","овсянка","пшено","манка","макароны","вермишель","спагетти","лапша",
        "чипсы","орехи","арахис","миндаль","фисташки","грецкий орех",
        "яблоки","бананы","апельсины","мандарины","груши","виноград","персики","абрикосы","сливы","киви","лимоны",
        "картофель","морковь","свекла","лук","чеснок","капуста","огурцы","помидоры","перец","баклажаны","кабачки","тыква",
        "укроп","петрушка","салат","шпинат","зелень",
        "сахар","соль","перец молотый","приправы","кетчуп","майонез","горчица",
        # однокоренные/синонимы
        "продукты","продукт","продуктывый","прод","еда","питание","бакалея","молочка","выпечка","овощи","фрукты"
    ],
    "Одежда": [
        "футболка","рубашка","кофта","свитер","толстовка","пиджак","жилет","пальто","куртка","плащ","шуба",
        "брюки","джинсы","шорты","юбка","платье","комбинезон","колготки","носки","гетры",
        "обувь","ботинки","туфли","кроссовки","кеды","сланцы","тапочки","сандалии",
        "одежда","шмот","шмотки","вещи","толстовки","толстовочка","кофточка"
    ],
    "Детские товары": [
        "подгузники","памперсы","соска","бутылочка","детская кроватка","коляска","детская одежда","детские ботинки","игрушки",
        "детская книга","детское питание","детская смесь","детский крем","пеленка","манеж","детское","ребенок","малыш"
    ],
    "Хозтовары": [
        "мусорные пакеты","губка","тряпка","ведро","швабра","метла","совок","щетка","лампочка","батарейки","зажигалка","пакеты","салфетки",
        "хозтовары","хоз","дом","домашние","фольга","пергамент","пленка пищевая"
    ],
    "Бытовая химия": [
        "стиральный порошок","кондиционер для белья","чистящее средство","средство для мытья посуды","отбеливатель",
        "доместос","фейри","санокс","антижир","химия","уборка","бытхимия","освежитель"
    ],
    "Лекарства": [
        "парацетамол","ибупрофен","аспирин","но-шпа","пластырь","мазь","капли","витамины","анальгин","цитрамон",
        "лекарства","таблетки","аптека","лекарство","термометр","сироп","спрей"
    ],
    "Авто": [
        "бензин","дизель","масло моторное","антифриз","омыватель","шины","аккумулятор","тормозная жидкость","автомойка",
        "авто","машина","автомобиль","транспорт","колодки","фильтр"
    ],
    "Строительство": [
        "цемент","кирпич","доска","гипсокартон","шпаклевка","краска","кисть","валик","гвозди","саморезы","шурупы","герметик",
        "строительство","ремонт","строймат","стройка","смесь","праймер","грунтовка","затирка"
    ],
    "Инструменты": [
        "отвертка","молоток","дрель","шуруповерт","болгарка","пила","рулетка","уровень","плоскогубцы","кусачки","степлер строительный","набор бит",
        "инструменты","инструмент","набор инструментов"
    ],
    "Электроника": [
        "телефон","ноутбук","планшет","монитор","клавиатура","мышь","наушники","зарядка","пауэрбанк","телевизор","смарт-часы","колонка",
        "электроника","гаджеты","техника","кабель","адаптер","роутер","флешка","ssd","hdd"
    ],
    "Канцтовары": [
        "ручка","карандаш","тетрадь","блокнот","маркер","степлер","скрепки","бумага","папка","ножницы","линейка",
        "канцелярия","канцтовары","канц","стикеры","клей карандаш","ластик"
    ],
    "Спорт": [
        "мяч","гантели","штанга","скакалка","коврик","велосипед","тренажер","форма","кроссовки спортивные","рюкзак спортивный",
        "спорт","спорттовары","тренировка","фитнес","эспандер","гантеля"
    ],
    "Здоровье": [
        "стоматолог","дантист","зубной","поликлиника","клиника","врач","прием","анализы","диагностика","мрт","кт",
        "медосмотр","медицинский","медцентр","медуслуги","массаж","физиотерапия","здоровье","реабилитация"
    ],
    "Подарки": [
        "подарок","подарочки","сувенир","букет","цветы","конфеты","шоколад","игрушка","подарочная карта","сертификат",
        "подарочный","дар","презент"
    ],
    "Развлечения": [
        "кино","кинтеатр","театр","концерт","бар","паб","кафе","ресторан","гулянка","гулянки","караоке","аттракцион",
        "вечеринка","клуб","досуг","развлечения","боулинг","бильярд","квест"
    ],
    "Коммуналка": [
        "коммуналка","кварплата","жкх","электроэнергия","свет","газ","вода","водоснабжение","отопление","мусор",
        "канализация","домофон","интернет","связь","телефон","айпи-тв","ip tv","кабельное","интеренет"
    ],
    "Кредит/Рассрочка": [
        "кредит","ипотека","рассрочка","платеж по кредиту","погашение","ежемесячный платеж","микрозайм","ломбард",
        "банк","проценты","переплата","эквайринг долг"
    ],
    "Прочее": [
        "подарок","сувенир","книга","журнал","газета","разное","прочее","непонятно","всякое","проч"
    ]
}

# 2) Нормализация текста
def normalize(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    t = t.replace("ё","е")
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"[^a-zа-я0-9\s\-_/\.]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# 3) Быстрый словарный матч (по подстроке любого ключа)
def dict_match_category(text_norm: str) -> str | None:
    for cat, words in CATEGORIES.items():
        for w in words:
            if w in text_norm:
                return cat
    return None

# 4) Простой фуззи-матч (char trigram overlap) без внешних зависимостей
def trigram_set(s: str) -> set[str]:
    s = f"  {s}  "
    return {s[i:i+3] for i in range(len(s)-2)}

def fuzzy_category(text_norm: str, threshold: float = 0.45) -> str | None:
    if not text_norm:
        return None
    best_cat, best_score = None, 0.0
    tset = trigram_set(text_norm)
    for cat, words in CATEGORIES.items():
        for w in words:
            wset = trigram_set(w)
            inter = len(tset & wset)
            union = len(tset | wset)
            score = inter / union if union else 0.0
            if score > best_score:
                best_score, best_cat = score, cat
    return best_cat if best_score >= threshold else None

# 5) ML-модель (char n-grams устойчивы к опечаткам)
vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3,5),
    min_df=1,
    max_features=40000
)
classifier = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

# Генерация обучающего набора из словаря + (опционально) TRAINING_DATA
BASE_TRAIN = []
for cat, words in CATEGORIES.items():
    for w in words:
        BASE_TRAIN.append((w, cat))

try:
    if isinstance(TRAINING_DATA, list) and TRAINING_DATA:
        BASE_TRAIN.extend(TRAINING_DATA)
except NameError:
    pass

def train_model(data):
    """
    Совместимость с существующим вызовом train_model(TRAINING_DATA):
    если data пустой — обучаемся на BASE_TRAIN.
    """
    use_data = data if (isinstance(data, list) and len(data) > 0) else BASE_TRAIN
    if not use_data:
        logger.warning("Нет данных для обучения модели. Модель не будет обучена.")
        return
    
    try:
        descriptions = [normalize(item[0]) for item in use_data]
        categories = [item[1] for item in use_data]

        # Обучаем модель
        X = vectorizer.fit_transform(descriptions)
        classifier.fit(X, categories)

        # Обновляем словарь категорий новыми примерами
        for description, category in use_data:
            if category in CATEGORIES:
                desc_lower = description.lower().strip()
                if desc_lower and desc_lower not in [w.lower() for w in CATEGORIES[category]]:
                    CATEGORIES[category].append(desc_lower)
                    logger.info(f"Добавлено в категорию '{category}': {desc_lower}")

        logger.info(f"Модель классификации (гибрид) успешно обучена на {len(use_data)} записях.")
    except Exception as e:
        logger.error(f"Ошибка при обучении модели: {e}")
        raise

# Обучаем (main() позже всё равно вызовет train_model(TRAINING_DATA))
train_model(BASE_TRAIN)

def classify_expense(description: str) -> str:
    """
    Возвращает категорию для расхода.
    Порядок: словарь → фуззи → ML → 'Прочее'
    """
    try:
        text_norm = normalize(description)

        # 1) словарь
        cat = dict_match_category(text_norm)
        if cat:
            return cat
        
        # 2) фуззи
        cat = fuzzy_category(text_norm)
        if cat:
            return cat
        
        # 3) ML
        if hasattr(classifier, "classes_") and len(getattr(classifier, "classes_", [])) > 0:
            vec = vectorizer.transform([text_norm])
            pred = classifier.predict(vec)[0]
            return pred

        # 4) fallback
        return "Прочее"
    except Exception as e:
        logger.error(f"Ошибка при классификации: {e}. Возвращаю 'Прочее'.")
        return "Прочее"

# --- Функции для работы с базой данных ---
def get_db_connection():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        logger.error(f"Ошибка подключения к БД: {e}")
        return None

def init_db():
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        # Удаляем user_id и family_id если есть
        for col in ['user_id', 'family_id']:
            try:
                cursor.execute(f"ALTER TABLE expenses DROP COLUMN {col};")
                conn.commit()
                logger.info(f"Столбец {col} успешно удален из таблицы expenses.")
            except psycopg2.errors.UndefinedColumn:
                conn.rollback()
            except Exception as e:
                conn.rollback()
                logger.error(f"Ошибка при удалении столбца {col}: {e}")
        
        # Создаем таблицу расходов
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS expenses (
                id SERIAL PRIMARY KEY,
                amount NUMERIC(10, 2) NOT NULL,
                description TEXT,
                category VARCHAR(100) NOT NULL,
                transaction_date TIMESTAMP WITH TIME ZONE NOT NULL
            );
        ''')
        
        # Создаем таблицу напоминаний
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS payment_reminders (
                id SERIAL PRIMARY KEY,
                title VARCHAR(200) NOT NULL,
                description TEXT,
                amount NUMERIC(10, 2) NOT NULL,
                start_date DATE NOT NULL,
                end_date DATE NOT NULL,
                reminder_10_days BOOLEAN DEFAULT FALSE,
                reminder_3_days BOOLEAN DEFAULT FALSE,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        ''')
        
        # Создаем таблицы для планирования бюджета
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS budget_plans (
                id SERIAL PRIMARY KEY,
                plan_month DATE NOT NULL UNIQUE,
                total_amount NUMERIC(12,2) NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                group_id INTEGER DEFAULT 1
            );
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS budget_plan_items (
                id SERIAL PRIMARY KEY,
                plan_id INTEGER NOT NULL REFERENCES budget_plans(id) ON DELETE CASCADE,
                category VARCHAR(100) NOT NULL,
                amount NUMERIC(12,2) NOT NULL,
                comment TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        ''')
        
        # Создаем таблицы для управления группами
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS groups (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) UNIQUE NOT NULL,
                admin_user_id INTEGER NOT NULL,
                max_members INTEGER DEFAULT 5,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                invitation_code VARCHAR(20) UNIQUE NOT NULL
            );
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS group_members (
                id SERIAL PRIMARY KEY,
                group_id INTEGER NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
                user_id INTEGER NOT NULL,
                phone VARCHAR(20) NOT NULL,
                role VARCHAR(20) DEFAULT 'member',
                joined_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        ''')
        
        # Добавляем group_id в существующие таблицы
        try:
            cursor.execute('ALTER TABLE expenses ADD COLUMN IF NOT EXISTS group_id INTEGER DEFAULT 1')
            cursor.execute('ALTER TABLE payment_reminders ADD COLUMN IF NOT EXISTS group_id INTEGER DEFAULT 1')
        except Exception as e:
            logger.info(f"Столбцы group_id уже существуют или не могут быть добавлены: {e}")
        
        # Создаем таблицы для системы управления пользователями (Railway/Cloud)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_folders (
                id SERIAL PRIMARY KEY,
                username VARCHAR(100) NOT NULL,
                user_id BIGINT,
                folder_name VARCHAR(100) NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                role VARCHAR(20) DEFAULT 'user',
                settings JSONB DEFAULT '{}',
                permissions JSONB DEFAULT '{}',
                UNIQUE(username, user_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_categories (
                id SERIAL PRIMARY KEY,
                user_id BIGINT,
                category_name VARCHAR(100) NOT NULL,
                keywords TEXT[] DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, category_name)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_settings (
                id SERIAL PRIMARY KEY,
                user_id BIGINT,
                setting_key VARCHAR(100) NOT NULL,
                setting_value JSONB,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, setting_key)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_logs (
                id SERIAL PRIMARY KEY,
                user_id BIGINT,
                log_level VARCHAR(20) NOT NULL,
                message TEXT NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_data (
                id SERIAL PRIMARY KEY,
                user_id BIGINT NOT NULL,
                data_type VARCHAR(50) NOT NULL,
                data_content JSONB NOT NULL,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, data_type)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_backups (
                id SERIAL PRIMARY KEY,
                user_id BIGINT NOT NULL,
                backup_name VARCHAR(100) NOT NULL,
                backup_data JSONB NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("База данных инициализирована (все таблицы проверены/созданы).")

def add_expense(amount, category, description, transaction_date, user_id=None):
    conn = get_db_connection()
    if not conn:
        return False
    try:
        cursor = conn.cursor()
        
        # Определяем group_id для пользователя
        group_id = 1  # По умолчанию
        if user_id:
            group_info = get_user_group(user_id)
            if group_info:
                group_id = group_info["id"]
        
        cursor.execute('''
            INSERT INTO expenses (amount, category, description, transaction_date, group_id)
            VALUES (%s, %s, %s, %s, %s)
        ''', (amount, category, description, transaction_date, group_id))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Ошибка при добавлении расхода: {e}")
        return False
    finally:
        conn.close()

def get_expense_by_id(expense_id):
    """Получить расход по ID"""
    conn = get_db_connection()
    if not conn:
        return None
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, amount, description, category, transaction_date
            FROM expenses WHERE id = %s
        ''', (expense_id,))
        return cursor.fetchone()
    except Exception as e:
        logger.error(f"Ошибка при получении расхода: {e}")
        return None
    finally:
        conn.close()

def update_expense_category(expense_id, new_category):
    """Обновить категорию расхода"""
    conn = get_db_connection()
    if not conn:
        return False
    try:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE expenses SET category = %s WHERE id = %s
        ''', (new_category, expense_id))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Ошибка при обновлении категории: {e}")
        return False
    finally:
        conn.close()

def update_expense_amount(expense_id: int, new_amount: float) -> bool:
    """Обновить сумму расхода"""
    conn = get_db_connection()
    if not conn:
        return False
    try:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE expenses SET amount = %s WHERE id = %s
        ''', (new_amount, expense_id))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Ошибка при обновлении суммы: {e}")
        return False
    finally:
        conn.close()

def get_recent_expenses(limit=10):
    """Получить последние расходы для исправления"""
    conn = get_db_connection()
    if not conn:
        return []
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, amount, description, category, transaction_date
            FROM expenses 
            ORDER BY transaction_date DESC 
            LIMIT %s
        ''', (limit,))
        return cursor.fetchall()
    except Exception as e:
        logger.error(f"Ошибка при получении расходов: {e}")
        return []
    finally:
        conn.close()

def get_all_expenses_for_training():
    """Получить все расходы для обучения модели"""
    conn = get_db_connection()
    if not conn:
        return []
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT description, category FROM expenses
        ''')
        return cursor.fetchall()
    except Exception as e:
        logger.error(f"Ошибка при получении данных для обучения: {e}")
        return []
    finally:
        conn.close()

def delete_expense(expense_id: int) -> bool:
    """Удалить расход по ID"""
    conn = get_db_connection()
    if not conn:
        return False
    try:
        cursor = conn.cursor()
        cursor.execute('''
            DELETE FROM expenses WHERE id = %s
        ''', (expense_id,))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Ошибка при удалении расхода: {e}")
        return False
    finally:
        conn.close()

# --- Функции для работы с напоминаниями ---
def add_payment_reminder(title, description, amount, start_date, end_date):
    """Добавить новое напоминание о платеже"""
    conn = get_db_connection()
    if not conn:
        return False
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO payment_reminders (title, description, amount, start_date, end_date)
            VALUES (%s, %s, %s, %s, %s)
        ''', (title, description, amount, start_date, end_date))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Ошибка при добавлении напоминания: {e}")
        return False
    finally:
        conn.close()

def get_all_active_reminders():
    """Получить все активные напоминания"""
    conn = get_db_connection()
    if not conn:
        return []
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, title, description, amount, start_date, end_date, 
                   reminder_10_days, reminder_3_days, created_at
            FROM payment_reminders 
            WHERE is_active = TRUE 
            ORDER BY end_date ASC
        ''')
        return cursor.fetchall()
    except Exception as e:
        logger.error(f"Ошибка при получении напоминаний: {e}")
        return []
    finally:
        conn.close()

def get_upcoming_reminders(days_ahead=30):
    """Получить напоминания, которые скоро истекают"""
    conn = get_db_connection()
    if not conn:
        return []
    try:
        cursor = conn.cursor()
        future_date = datetime.now().date() + timedelta(days=days_ahead)
        cursor.execute('''
            SELECT id, title, description, amount, start_date, end_date, 
                   reminder_10_days, reminder_3_days
            FROM payment_reminders 
            WHERE is_active = TRUE AND end_date <= %s
            ORDER BY end_date ASC
        ''', (future_date,))
        return cursor.fetchall()
    except Exception as e:
        logger.error(f"Ошибка при получении предстоящих напоминаний: {e}")
        return []
    finally:
        conn.close()

def mark_reminder_sent(reminder_id, reminder_type):
    """Отметить, что напоминание было отправлено"""
    conn = get_db_connection()
    if not conn:
        return False
    try:
        cursor = conn.cursor()
        if reminder_type == '10_days':
            cursor.execute('''
                UPDATE payment_reminders SET reminder_10_days = TRUE WHERE id = %s
            ''', (reminder_id,))
        elif reminder_type == '3_days':
            cursor.execute('''
                UPDATE payment_reminders SET reminder_3_days = TRUE WHERE id = %s
            ''', (reminder_id,))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Ошибка при обновлении статуса напоминания: {e}")
        return False
    finally:
        conn.close()

def deactivate_expired_reminder(reminder_id):
    """Деактивировать истекшее напоминание"""
    conn = get_db_connection()
    if not conn:
        return False
    try:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE payment_reminders SET is_active = FALSE WHERE id = %s
        ''', (reminder_id,))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Ошибка при деактивации напоминания: {e}")
        return False
    finally:
        conn.close()

def delete_reminder(reminder_id):
    """Удалить напоминание"""
    conn = get_db_connection()
    if not conn:
        return False
    try:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM payment_reminders WHERE id = %s', (reminder_id,))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Ошибка при удалении напоминания: {e}")
        return False
    finally:
        conn.close()

async def check_and_send_reminders(application):
    """Проверка и отправка автоматических напоминаний"""
    try:
        reminders = get_upcoming_reminders(15)  # Проверяем на 15 дней вперед
        current_date = datetime.now().date()
        
        for reminder in reminders:
            rem_id, title, desc, amount, start_date, end_date, sent_10, sent_3 = reminder
            days_left = (end_date - current_date).days
            
            # Проверяем, нужно ли отправить напоминание за 10 дней
            if days_left == 10 and not sent_10:
                message_text = f"⚠️ НАПОМИНАНИЕ О ПЛАТЕЖЕ!\n\n"
                message_text += f"📋 {title}\n"
                if desc:
                    message_text += f"📝 {desc}\n"
                message_text += f"💰 Сумма: {amount:.2f} Тг\n"
                message_text += f"📅 Срок действия истекает: {end_date.strftime('%d.%m.%Y')}\n"
                message_text += f"⏰ Осталось дней: {days_left}\n\n"
                message_text += f"💡 Не забудьте оплатить вовремя!"
                
                # Отправляем напоминание всем активным чатам
                # В реальном боте здесь нужно получить список пользователей
                # Пока что просто логируем
                logger.info(f"Отправлено напоминание за 10 дней: {title}")
                mark_reminder_sent(rem_id, '10_days')
            
            # Проверяем, нужно ли отправить напоминание за 3 дня
            elif days_left == 3 and not sent_3:
                message_text = f"🚨 СРОЧНОЕ НАПОМИНАНИЕ О ПЛАТЕЖЕ!\n\n"
                message_text += f"📋 {title}\n"
                if desc:
                    message_text += f"📝 {desc}\n"
                message_text += f"💰 Сумма: {amount:.2f} Тг\n"
                message_text += f"📅 Срок действия истекает: {end_date.strftime('%d.%m.%Y')}\n"
                message_text += f"⏰ Осталось дней: {days_left}\n\n"
                message_text += f"🔥 Оплатите сегодня, чтобы не было проблем!"
                
                logger.info(f"Отправлено срочное напоминание за 3 дня: {title}")
                mark_reminder_sent(rem_id, '3_days')
            
            # Деактивируем истекшие напоминания
            elif days_left < 0:
                deactivate_expired_reminder(rem_id)
                logger.info(f"Деактивировано истекшее напоминание: {title}")
                
    except Exception as e:
        logger.error(f"Ошибка при проверке напоминаний: {e}")

# --- ЗАЩИТА БЛОКОВ ---
# Константы для защиты блоков от несанкционированных изменений
BLOCK_PROTECTION = {
    "reports": True,
    "corrections": True, 
    "reminders": True,
    "planning": True,
    "analytics": True,
    "expenses": True,
    "training": True
}

def is_block_protected(block_name: str) -> bool:
    """Проверяет, защищен ли блок от изменений"""
    return BLOCK_PROTECTION.get(block_name, True)

def validate_block_access(block_name: str, user_id: int) -> bool:
    """Проверяет доступ пользователя к блоку (базовая защита)"""
    if not is_block_protected(block_name):
        return False
    
    # Проверяем, авторизован ли пользователь
    if not is_user_authorized(user_id):
        logger.info(f"Пользователь {user_id} не авторизован для доступа к блоку {block_name}")
        return False
    
    # В будущем здесь можно добавить проверку прав пользователя
    return True

# --- UI (User Interface) ---
def get_main_menu_keyboard():
    keyboard = [
        [KeyboardButton("💸 Добавить расход"), KeyboardButton("📊 Отчеты")],
        [KeyboardButton("🔧 Исправить категории"), KeyboardButton("📚 Обучить модель")],
        [KeyboardButton("⏰ Напоминания"), KeyboardButton("📅 Планирование")],
        [KeyboardButton("📈 Аналитика"), KeyboardButton("👥 Управление группой")]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

def get_admin_menu_keyboard():
    """Клавиатура для администратора"""
    keyboard = [
        [KeyboardButton("👥 Добавить пользователя"), KeyboardButton("📋 Список пользователей")],
        [KeyboardButton("📁 Управление папками"), KeyboardButton("🔧 Роли пользователей")],
        [KeyboardButton("📊 Статистика системы"), KeyboardButton("🔙 Главное меню")]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

def get_report_period_keyboard():
    keyboard = [
        [KeyboardButton("Сегодня"), KeyboardButton("Неделя")],
        [KeyboardButton("Месяц"), KeyboardButton("Год")]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)

def get_categories_keyboard():
    """Клавиатура с категориями для исправления"""
    categories = list(CATEGORIES.keys())
    # Разбиваем на ряды по 2 кнопки
    keyboard = []
    for i in range(0, len(categories), 2):
        row = [KeyboardButton(categories[i])]
        if i + 1 < len(categories):
            row.append(KeyboardButton(categories[i + 1]))
        keyboard.append(row)
    # Добавляем кнопку для создания новой категории
    keyboard.append([KeyboardButton("➕ Добавить новую категорию")])
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)

def get_categories_keyboard_with_done():
    """Клавиатура с категориями + кнопка Готово (для планирования)"""
    categories = list(CATEGORIES.keys())
    keyboard = []
    for i in range(0, len(categories), 2):
        row = [KeyboardButton(categories[i])]
        if i + 1 < len(categories):
            row.append(KeyboardButton(categories[i + 1]))
        keyboard.append(row)
    # Добавляем кнопку для создания новой категории
    keyboard.append([KeyboardButton("➕ Добавить новую категорию")])
    keyboard.append([KeyboardButton("Готово")])
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

async def manual_training_fallback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await manual_training(update, context)
    return ConversationHandler.END

# --- Команды бота ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отвечает на команду /start."""
    user_id = update.effective_user.id
    
    # Проверяем, является ли пользователь администратором
    users_data = load_authorized_users()
    is_admin = user_id == users_data.get("admin")
    
    if is_admin:
        # Показываем админ-меню
        await update.message.reply_text(
            "🔐 Админ-панель\n\nВыберите действие:",
            reply_markup=get_admin_menu_keyboard()
        )
        return
    
    # Проверяем, авторизован ли пользователь
    if not is_user_authorized(user_id):
        # Пользователь не авторизован - просим ввести username
        await update.message.reply_text(
            "🔐 Добро пожаловать!\n\n"
            "Для доступа к боту необходимо ввести ваше имя.\n"
            "👤 Введите ваше имя:",
            reply_markup=ReplyKeyboardMarkup([["🔙 Отмена"]], resize_keyboard=True)
        )
        context.user_data['auth_state'] = 'waiting_for_username'
        return
    
    # Пользователь авторизован - показываем главное меню
    await update.message.reply_text(
        "Привет! Я твой помощник по учету расходов. Выбери опцию ниже:",
        reply_markup=get_main_menu_keyboard()
    )

# --- АДМИН-ФУНКЦИИ ---
async def admin_menu_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик админ-меню"""
    user_id = update.effective_user.id
    text = update.message.text
    
    # Проверяем, является ли пользователь администратором
    users_data = load_authorized_users()
    if user_id != users_data.get("admin"):
        await update.message.reply_text(
            "❌ Доступ запрещен. Только администратор может использовать эту функцию.",
            reply_markup=get_main_menu_keyboard()
        )
        return
    
    if text == "👥 Добавить пользователя":
        await update.message.reply_text(
            "👤 Введите имя нового пользователя:\n\n"
            "Например: Иван Петров, Мария Сидорова\n\n"
            "⚠️ Имя должно быть уникальным и не повторяться",
            reply_markup=ReplyKeyboardMarkup([["🔙 Назад"]], resize_keyboard=True)
        )
        return 'waiting_for_username'
    
    elif text == "📋 Список пользователей":
        users = get_authorized_users_list()
        if not users:
            await update.message.reply_text(
                "📋 Список пользователей пуст.",
                reply_markup=get_admin_menu_keyboard()
            )
            return
        
        users_text = "📋 Список авторизованных пользователей:\n\n"
        for i, user in enumerate(users, 1):
            username = user.get("username", "Не указано")
            added_date = user.get("added_date", "Не указана")
            status = user.get("status", "Неизвестен")
            telegram_id = user.get("telegram_id", "Не привязан")
            role = user.get("role", "user")
            role_name = USER_ROLES.get(role, role)
            folder_name = user.get("folder_name", "Не задана")
            
            users_text += f"{i}. 👤 {username}\n"
            users_text += f"   🆔 Telegram ID: {telegram_id}\n"
            users_text += f"   📅 Добавлен: {added_date[:10]}\n"
            users_text += f"   ✅ Статус: {status}\n"
            users_text += f"   🔧 Роль: {role_name}\n"
            users_text += f"   📁 Папка: {folder_name}\n\n"
        
        await update.message.reply_text(
            users_text,
            reply_markup=get_admin_menu_keyboard()
        )
        return
    
    elif text == "📁 Управление папками":
        await admin_folder_management(update, context)
        return
    
    elif text == "🔧 Роли пользователей":
        await admin_roles_management(update, context)
        return
    
    elif text == "📊 Статистика системы":
        await admin_system_stats(update, context)
        return
    
    elif text == "🔙 Главное меню":
        await update.message.reply_text(
            "Главное меню:",
            reply_markup=get_main_menu_keyboard()
        )
        return

async def admin_username_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    """Обработчик ввода имени пользователя для добавления"""
    user_id = update.effective_user.id
    text = update.message.text
    
    # Проверяем, является ли пользователь администратором
    users_data = load_authorized_users()
    if user_id != users_data.get("admin"):
        await update.message.reply_text(
            "❌ Доступ запрещен. Только администратор может использовать эту функцию.",
            reply_markup=get_main_menu_keyboard()
        )
        return ConversationHandler.END
    
    username = text.strip()
    
    # Проверяем, что имя не пустое и содержит только буквы, цифры и пробелы
    if not username or len(username) < 2:
        await update.message.reply_text(
            "❌ Имя пользователя должно содержать минимум 2 символа.\n\n"
            "Попробуйте еще раз:",
            reply_markup=ReplyKeyboardMarkup([["🔙 Назад"]], resize_keyboard=True)
        )
        return 'waiting_for_username'
    
    # Сохраняем имя пользователя в контексте
    context.user_data['new_username'] = username
    
    # Запрашиваем название папки
    await update.message.reply_text(
        f"👤 Пользователь: {username}\n\n"
        "📁 Введите уникальное название для папки пользователя\n"
        "(например: 'my_finances', 'personal_budget'):",
        reply_markup=ReplyKeyboardMarkup([["🔙 Назад"]], resize_keyboard=True)
    )
    
    return "waiting_folder_name"

async def admin_back_to_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    """Возврат в админское меню"""
    await update.message.reply_text(
        "Админ-меню:",
        reply_markup=get_admin_menu_keyboard()
    )
    return ConversationHandler.END

async def admin_folder_name_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    """Обработчик ввода названия папки для нового пользователя"""
    user_id = update.effective_user.id
    text = update.message.text
    
    if text == "🔙 Назад":
        await update.message.reply_text(
            "👥 Введите имя пользователя для добавления:",
            reply_markup=ReplyKeyboardMarkup([["🔙 Назад"]], resize_keyboard=True)
        )
        return "waiting_for_username"
    
    folder_name = text.strip()
    username = context.user_data.get('new_username')
    
    # Запрашиваем роль пользователя
    await update.message.reply_text(
        f"👤 Пользователь: {username}\n"
        f"📁 Папка: {folder_name}\n\n"
        "🔧 Выберите роль пользователя:",
        reply_markup=ReplyKeyboardMarkup([
            [KeyboardButton("👤 Обычный пользователь"), KeyboardButton("🛡️ Модератор")],
            [KeyboardButton("🔙 Назад")]
        ], resize_keyboard=True)
    )
    
    context.user_data['new_folder_name'] = folder_name
    return "waiting_role"

async def admin_role_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    """Обработчик выбора роли для нового пользователя"""
    user_id = update.effective_user.id
    text = update.message.text
    
    if text == "🔙 Назад":
        await update.message.reply_text(
            "📁 Введите уникальное название для папки пользователя:",
            reply_markup=ReplyKeyboardMarkup([["🔙 Назад"]], resize_keyboard=True)
        )
        return "waiting_folder_name"
    
    username = context.user_data.get('new_username')
    folder_name = context.user_data.get('new_folder_name')
    
    # Определяем роль
    if text == "👤 Обычный пользователь":
        role = "user"
    elif text == "🛡️ Модератор":
        role = "moderator"
    else:
        await update.message.reply_text(
            "❌ Неверный выбор роли. Попробуйте снова:",
            reply_markup=ReplyKeyboardMarkup([
                [KeyboardButton("👤 Обычный пользователь"), KeyboardButton("🛡️ Модератор")],
                [KeyboardButton("🔙 Назад")]
            ], resize_keyboard=True)
        )
        return "waiting_role"
    
    # Добавляем пользователя
    logger.info(f"Попытка добавить пользователя '{username}' с папкой '{folder_name}' и ролью '{role}'")
    # Используем временный ID 0, который будет обновлен при первом входе пользователя
    success, message = add_authorized_user(username, 0, folder_name, role)
    
    if success:
        logger.info(f"Пользователь '{username}' успешно добавлен")
        await update.message.reply_text(
            f"✅ Пользователь успешно добавлен!\n\n"
            f"👤 Имя: {username}\n"
            f"📁 Папка: {folder_name}\n"
            f"🔧 Роль: {USER_ROLES.get(role, role)}\n"
            f"📅 Дата: {datetime.now().strftime('%d.%m.%Y')}\n\n"
            "Пользователь может теперь запустить бота командой /start",
            reply_markup=get_admin_menu_keyboard()
        )
    else:
        logger.error(f"Ошибка при добавлении пользователя '{username}': {message}")
        await update.message.reply_text(
            f"❌ {message}\n\n"
            "Попробуйте другое имя:",
            reply_markup=ReplyKeyboardMarkup([["🔙 Назад"]], resize_keyboard=True)
        )
    
    # Очищаем контекст
    context.user_data.pop('new_username', None)
    context.user_data.pop('new_folder_name', None)
    
    return ConversationHandler.END

async def admin_folder_management(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Управление папками пользователей (из базы данных)"""
    user_id = update.effective_user.id
    
    # Проверяем, является ли пользователь администратором
    users_data = load_authorized_users()
    if user_id != users_data.get("admin"):
        await update.message.reply_text(
            "❌ Доступ запрещен. Только администратор может использовать эту функцию.",
            reply_markup=get_main_menu_keyboard()
        )
        return
    
    try:
        conn = get_db_connection()
        if not conn:
            await update.message.reply_text(
                "❌ Ошибка подключения к базе данных.",
                reply_markup=get_admin_menu_keyboard()
            )
            return
        
        cursor = conn.cursor()
        
        # Получаем список всех папок пользователей из БД
        cursor.execute('''
            SELECT username, user_id, folder_name, role, created_at,
                   (SELECT COUNT(*) FROM user_data WHERE user_id = uf.user_id) as data_count
            FROM user_folders uf
            ORDER BY created_at DESC
        ''')
        
        folders = cursor.fetchall()
        conn.close()
        
        if not folders:
            await update.message.reply_text(
                "📁 Папки пользователей не найдены в базе данных.",
                reply_markup=get_admin_menu_keyboard()
            )
            return
        
        # Формируем список папок
        folders_text = "📁 Список папок пользователей (из БД):\n\n"
        for i, folder in enumerate(folders, 1):
            username, user_id, folder_name, role, created_at, data_count = folder
            role_name = USER_ROLES.get(role, role)
            
            folders_text += f"{i}. 👤 {username}\n"
            folders_text += f"   📁 Папка: {folder_name}\n"
            folders_text += f"   🔧 Роль: {role_name}\n"
            folders_text += f"   📅 Создана: {created_at.strftime('%d.%m.%Y %H:%M')}\n"
            folders_text += f"   📊 Данных: {data_count} записей\n\n"
        
        await update.message.reply_text(
            folders_text,
            reply_markup=get_admin_menu_keyboard()
        )
        
    except Exception as e:
        logger.error(f"Ошибка при получении списка папок: {e}")
        await update.message.reply_text(
            f"❌ Ошибка при получении списка папок: {e}",
            reply_markup=get_admin_menu_keyboard()
        )

async def admin_roles_management(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Управление ролями пользователей"""
    user_id = update.effective_user.id
    
    # Проверяем, является ли пользователь администратором
    users_data = load_authorized_users()
    if user_id != users_data.get("admin"):
        await update.message.reply_text(
            "❌ Доступ запрещен. Только администратор может использовать эту функцию.",
            reply_markup=get_main_menu_keyboard()
        )
        return
    
    users = get_authorized_users_list()
    if not users:
        await update.message.reply_text(
            "👥 Пользователи не найдены.",
            reply_markup=get_admin_menu_keyboard()
        )
        return
    
    # Формируем список пользователей с ролями
    roles_text = "🔧 Управление ролями пользователей:\n\n"
    for user in users:
        role = user.get('role', 'user')
        role_name = USER_ROLES.get(role, role)
        status = user.get('status', 'unknown')
        folders_text = f"📁 {user.get('folder_name', 'Не задана')}" if user.get('folder_name') else "📁 Не создана"
        
        roles_text += f"👤 {user.get('username', 'Неизвестно')}\n"
        roles_text += f"   🔧 Роль: {role_name}\n"
        roles_text += f"   📊 Статус: {status}\n"
        roles_text += f"   {folders_text}\n\n"
    
    await update.message.reply_text(
        roles_text,
        reply_markup=get_admin_menu_keyboard()
    )

async def admin_system_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Статистика системы (из базы данных)"""
    user_id = update.effective_user.id
    
    # Проверяем, является ли пользователь администратором
    users_data = load_authorized_users()
    if user_id != users_data.get("admin"):
        await update.message.reply_text(
            "❌ Доступ запрещен. Только администратор может использовать эту функцию.",
            reply_markup=get_main_menu_keyboard()
        )
        return
    
    try:
        conn = get_db_connection()
        if not conn:
            await update.message.reply_text(
                "❌ Ошибка подключения к базе данных.",
                reply_markup=get_admin_menu_keyboard()
            )
            return
        
        cursor = conn.cursor()
        
        # Статистика пользователей
        cursor.execute('SELECT COUNT(*) FROM user_folders')
        total_folders = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT user_id) FROM user_folders')
        unique_users = cursor.fetchone()[0]
        
        # Статистика по ролям
        cursor.execute('''
            SELECT role, COUNT(*) 
            FROM user_folders 
            GROUP BY role
        ''')
        role_stats = dict(cursor.fetchall())
        
        # Статистика данных
        cursor.execute('SELECT COUNT(*) FROM user_data')
        total_data_records = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM user_backups')
        total_backups = cursor.fetchone()[0]
        
        # Статистика расходов
        cursor.execute('SELECT COUNT(*) FROM expenses')
        total_expenses = cursor.fetchone()[0]
        
        # Статистика напоминаний
        cursor.execute('SELECT COUNT(*) FROM reminders')
        total_reminders = cursor.fetchone()[0]
        
        conn.close()
        
        # Формируем отчет
        stats_text = "📊 Статистика системы (Railway/Cloud):\n\n"
        stats_text += f"👥 Пользователи:\n"
        stats_text += f"   📈 Всего папок: {total_folders}\n"
        stats_text += f"   👤 Уникальных пользователей: {unique_users}\n\n"
        
        stats_text += f"🔧 Роли:\n"
        for role, count in role_stats.items():
            role_name = USER_ROLES.get(role, role)
            stats_text += f"   {role_name}: {count}\n"
        
        stats_text += f"\n📊 Данные:\n"
        stats_text += f"   💾 Записей данных: {total_data_records}\n"
        stats_text += f"   💰 Расходов: {total_expenses}\n"
        stats_text += f"   ⏰ Напоминаний: {total_reminders}\n"
        stats_text += f"   💾 Резервных копий: {total_backups}\n\n"
        
        stats_text += f"☁️ Платформа: Railway (PostgreSQL)\n"
        stats_text += f"🕐 Доступность: 24/7\n"
        
        await update.message.reply_text(
            stats_text,
            reply_markup=get_admin_menu_keyboard()
        )
        
    except Exception as e:
        logger.error(f"Ошибка при получении статистики: {e}")
        await update.message.reply_text(
            f"❌ Ошибка при получении статистики: {e}",
            reply_markup=get_admin_menu_keyboard()
        )

# --- ОБРАБОТЧИК АУТЕНТИФИКАЦИИ ---
async def auth_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик аутентификации пользователей"""
    user_id = update.effective_user.id
    text = update.message.text
    
    if text == "🔙 Отмена":
        await update.message.reply_text(
            "❌ Доступ к боту отменен. Обратитесь к администратору.",
            reply_markup=ReplyKeyboardRemove()
        )
        context.user_data.pop('auth_state', None)
        return
    
    # Проверяем, что пользователь находится в состоянии ожидания username
    if context.user_data.get('auth_state') == 'waiting_for_username':
        username = text.strip()
        
        if len(username) < 2:
            await update.message.reply_text(
                "❌ Имя должно содержать минимум 2 символа.\n\n"
                "Попробуйте еще раз:",
                reply_markup=ReplyKeyboardMarkup([["🔙 Отмена"]], resize_keyboard=True)
            )
            return
        
        # Проверяем, есть ли username в списке авторизованных
        if is_username_authorized(username):
            logger.info(f"Пользователь {user_id} авторизован по имени '{username}'")
            
            # Обновляем telegram_id для этого пользователя
            users_data = load_authorized_users()
            for user in users_data.get("users", []):
                if user.get("username") == username:
                    user["telegram_id"] = user_id
                    save_authorized_users(users_data)
                    logger.info(f"Обновлен telegram_id для пользователя '{username}': {user_id}")
                    break
            
            await update.message.reply_text(
                "✅ Ваше имя найдено в списке авторизованных пользователей!\n\n"
                "Теперь у вас есть доступ к боту!",
                reply_markup=get_main_menu_keyboard()
            )
            context.user_data.pop('auth_state', None)
            return
        else:
            await update.message.reply_text(
                "❌ Ваше имя не найдено в списке авторизованных пользователей.\n\n"
                "Обратитесь к администратору для добавления в список:",
                reply_markup=ReplyKeyboardMarkup([["🔙 Отмена"]], resize_keyboard=True)
            )
            return
    
    # Неизвестное состояние
    await update.message.reply_text(
        "❌ Ошибка авторизации. Попробуйте еще раз.",
        reply_markup=ReplyKeyboardRemove()
    )
    context.user_data.pop('auth_state', None)
    return ConversationHandler.END

# --- УПРАВЛЕНИЕ ГРУППОЙ ---
async def group_management_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Меню управления группой"""
    user_id = update.effective_user.id
    
    # Проверяем, находится ли пользователь в группе
    if not is_user_in_group(user_id):
        await update.message.reply_text(
            "❌ Вы не состоите ни в одной группе.",
            reply_markup=get_main_menu_keyboard()
        )
        return
    
    group_info = get_user_group(user_id)
    if not group_info:
        await update.message.reply_text(
            "❌ Ошибка при получении информации о группе.",
            reply_markup=get_main_menu_keyboard()
        )
        return
    
    # Проверяем, является ли пользователь админом группы
    is_group_admin = group_info["admin_user_id"] == user_id
    
    if is_group_admin:
        # Меню для админа группы
        keyboard = [
            [KeyboardButton("👥 Участники группы"), KeyboardButton("🔑 Код приглашения")],
            [KeyboardButton("📊 Статистика группы"), KeyboardButton("🔙 Главное меню")]
        ]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
        
        await update.message.reply_text(
            f"👥 Управление группой '{group_info['name']}'\n\n"
            "Вы являетесь администратором этой группы.\n\n"
            "Выберите действие:",
            reply_markup=reply_markup
        )
    else:
        # Меню для обычного участника
        keyboard = [
            [KeyboardButton("👥 Участники группы"), KeyboardButton("🔙 Главное меню")]
        ]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
        
        await update.message.reply_text(
            f"👥 Группа '{group_info['name']}'\n\n"
            "Выберите действие:",
            reply_markup=reply_markup
        )
    
    context.user_data['group_management_state'] = 'menu'

async def group_management_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик управления группой"""
    user_id = update.effective_user.id
    text = update.message.text
    
    if text == "🔙 Главное меню":
        await update.message.reply_text(
            "Главное меню:",
            reply_markup=get_main_menu_keyboard()
        )
        context.user_data.pop('group_management_state', None)
        return
    
    # Проверяем, находится ли пользователь в группе
    if not is_user_in_group(user_id):
        await update.message.reply_text(
            "❌ Вы не состоите ни в одной группе.",
            reply_markup=get_main_menu_keyboard()
        )
        context.user_data.pop('group_management_state', None)
        return
    
    group_info = get_user_group(user_id)
    if not group_info:
        await update.message.reply_text(
            "❌ Ошибка при получении информации о группе.",
            reply_markup=get_main_menu_keyboard()
        )
        context.user_data.pop('group_management_state', None)
        return
    
    is_group_admin = group_info["admin_user_id"] == user_id
    
    if text == "👥 Участники группы":
        members = get_group_members(group_info["id"])
        if not members:
            await update.message.reply_text(
                "❌ Ошибка при получении списка участников.",
                reply_markup=get_main_menu_keyboard()
            )
            return
        
        members_text = f"👥 Участники группы '{group_info['name']}':\n\n"
        for i, member in enumerate(members, 1):
            role_emoji = "👑" if member["role"] == "admin" else "👤"
            members_text += f"{i}. {role_emoji} {member['phone']}\n"
            members_text += f"   🆔 ID: {member['user_id']}\n"
            members_text += f"   📅 Присоединился: {member['joined_at'].strftime('%d.%m.%Y') if member['joined_at'] else 'Неизвестно'}\n\n"
        
        await update.message.reply_text(
            members_text,
            reply_markup=get_main_menu_keyboard()
        )
        context.user_data.pop('group_management_state', None)
        return
    
    elif text == "🔑 Код приглашения" and is_group_admin:
        await update.message.reply_text(
            f"🔑 Код приглашения для группы '{group_info['name']}':\n\n"
            f"📱 {group_info['invitation_code']}\n\n"
            "📤 Отправьте этот код членам семьи для присоединения к группе.\n\n"
            "⚠️ Код действителен только для вашей группы.",
            reply_markup=get_main_menu_keyboard()
        )
        context.user_data.pop('group_management_state', None)
        return
    
    elif text == "📊 Статистика группы" and is_group_admin:
        # Здесь можно добавить статистику по группе
        await update.message.reply_text(
            f"📊 Статистика группы '{group_info['name']}'\n\n"
            "Функция в разработке.",
            reply_markup=get_main_menu_keyboard()
        )
        context.user_data.pop('group_management_state', None)
        return
    
    else:
        await update.message.reply_text(
            "❌ Неизвестная команда. Используйте кнопки меню.",
            reply_markup=get_main_menu_keyboard()
        )
        context.user_data.pop('group_management_state', None)
        return

async def report_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    
    # Проверяем защиту блока отчетов
    if not validate_block_access("reports", user_id):
        await update.message.reply_text(
            "❌ Доступ к отчетам ограничен.",
            reply_markup=get_main_menu_keyboard()
        )
        return ConversationHandler.END
    
    await update.message.reply_text(
        "За какой период вы хотите отчет?",
        reply_markup=get_report_period_keyboard()
    )
    return PERIOD_CHOICE_STATE  # Важно! Переводим в состояние выбора периода

async def correction_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Меню исправления категорий"""
    user_id = update.effective_user.id
    
    # Проверяем защиту блока исправлений
    if not validate_block_access("corrections", user_id):
        await update.message.reply_text(
            "❌ Доступ к исправлениям ограничен.",
            reply_markup=get_main_menu_keyboard()
        )
        return ConversationHandler.END
    
    keyboard = [
        [KeyboardButton("1️⃣ Исправить расход")],
        [KeyboardButton("2️⃣ Удалить расход")],
        [KeyboardButton("3️⃣ Назад")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)
    
    await update.message.reply_text(
        "🔧 Выберите действие:\n\n"
        "1️⃣ Исправить расход - изменить категорию или сумму\n"
        "2️⃣ Удалить расход - удалить запись из базы\n"
        "3️⃣ Назад - вернуться в главное меню",
        reply_markup=reply_markup
    )
    return CORRECTION_MENU_STATE

async def correction_menu_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора в меню исправления расходов"""
    choice = update.message.text.strip()
    
    if choice == "1️⃣ Исправить расход":
        # Показываем список расходов для исправления
        expenses = get_recent_expenses(10)
        if not expenses:
            await update.message.reply_text(
                "Нет расходов для исправления. Сначала добавьте несколько расходов.",
                reply_markup=get_main_menu_keyboard()
            )
            return ConversationHandler.END
        
        # Формируем список расходов для выбора
        expenses_text = "Выберите расход для исправления категории:\n\n"
        for i, (exp_id, amount, desc, cat, date) in enumerate(expenses, 1):
            date_str = date.strftime("%d.%m.%Y") if date else "Неизвестно"
            expenses_text += f"{i}. {desc} - {amount} Тг ({cat}) - {date_str}\n"
        
        # Сохраняем расходы в контексте
        context.user_data['expenses_to_correct'] = expenses
        
        await update.message.reply_text(
            expenses_text + "\nВведите номер расхода (1-10):",
            reply_markup=ReplyKeyboardRemove()
        )
        return EXPENSE_CHOICE_STATE
        
    elif choice == "2️⃣ Удалить расход":
        # Показываем список расходов для удаления
        expenses = get_recent_expenses(10)
        if not expenses:
            await update.message.reply_text(
                "Нет расходов для удаления. Сначала добавьте несколько расходов.",
                reply_markup=get_main_menu_keyboard()
            )
            return ConversationHandler.END
        
        # Формируем список расходов для выбора
        expenses_text = "Выберите расход для удаления:\n\n"
        for i, (exp_id, amount, desc, cat, date) in enumerate(expenses, 1):
            date_str = date.strftime("%d.%m.%Y") if date else "Неизвестно"
            expenses_text += f"{i}. {desc} - {amount} Тг ({cat}) - {date_str}\n"
        
        # Сохраняем расходы в контексте
        context.user_data['expenses_to_delete'] = expenses
        
        await update.message.reply_text(
            expenses_text + "\nВведите номер расхода для удаления (1-10):",
            reply_markup=ReplyKeyboardRemove()
        )
        return EXPENSE_DELETE_STATE
        
    elif choice == "3️⃣ Назад":
        await update.message.reply_text(
            "Возвращаемся в главное меню",
            reply_markup=get_main_menu_keyboard()
        )
        return ConversationHandler.END
    
    else:
        await update.message.reply_text(
            "Пожалуйста, выберите один из предложенных вариантов:",
            reply_markup=get_main_menu_keyboard()
        )
        return ConversationHandler.END

async def expense_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Выбор расхода для исправления"""
    try:
        choice = int(update.message.text)
        expenses = context.user_data.get('expenses_to_correct', [])
        
        if choice < 1 or choice > len(expenses):
            await update.message.reply_text(
                f"Пожалуйста, введите число от 1 до {len(expenses)}",
                reply_markup=get_main_menu_keyboard()
            )
            return ConversationHandler.END
        
        # Сохраняем выбранный расход
        selected_expense = expenses[choice - 1]
        context.user_data['selected_expense'] = selected_expense
        
        exp_id, amount, desc, cat, date = selected_expense
        date_str = date.strftime("%d.%m.%Y") if date else "Неизвестно"
        
        await update.message.reply_text(
            f"Выбран расход:\n"
            f"📝 {desc}\n"
            f"💰 {amount} Тг\n"
            f"🏷️ Текущая категория: {cat}\n"
            f"📅 {date_str}\n\n"
            f"Выберите новую категорию:",
            reply_markup=get_categories_keyboard()
        )
        return CATEGORY_CHOICE_STATE
        
    except ValueError:
        await update.message.reply_text(
            "Пожалуйста, введите число.",
            reply_markup=get_main_menu_keyboard()
        )
        return ConversationHandler.END

async def category_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Выбор новой категории для расхода"""
    new_category = update.message.text.strip()
    
    # Проверяем специальную кнопку для создания новой категории
    if new_category == "➕ Добавить новую категорию":
        await update.message.reply_text(
            "Введите название новой категории:",
            reply_markup=ReplyKeyboardRemove()
        )
        return CUSTOM_CATEGORY_STATE
    
    # Проверяем, что категория не пустая
    if not new_category:
        await update.message.reply_text(
            "Название категории не может быть пустым. Попробуйте снова:",
            reply_markup=get_categories_keyboard()
        )
        return CATEGORY_CHOICE_STATE
    
    # Проверяем, существует ли категория в списке
    if new_category not in CATEGORIES:
        await update.message.reply_text(
            "Пожалуйста, выберите категорию из предложенных или нажмите '➕ Добавить новую категорию':",
            reply_markup=get_categories_keyboard()
        )
        return CATEGORY_CHOICE_STATE
    
    # Получаем выбранный расход
    selected_expense = context.user_data.get('selected_expense')
    if not selected_expense:
        await update.message.reply_text(
            "Ошибка: расход не найден. Попробуйте снова.",
            reply_markup=get_main_menu_keyboard()
        )
        return ConversationHandler.END
    
    exp_id, amount, desc, old_cat, date = selected_expense
    
    # Обновляем категорию в базе данных
    if update_expense_category(exp_id, new_category):
        context.user_data['selected_expense'] = (exp_id, amount, desc, new_category, date)
        await update.message.reply_text(
            f"✅ Категория обновлена!\n\n"
            f"📝 {desc}\n"
            f"🏷️ Новая категория: {new_category}\n"
            f"💰 Сумма: {amount} Тг\n\n"
            f"Теперь введите новую сумму (например: 1500.50) или отправьте текущую сумму без изменений:",
            reply_markup=ReplyKeyboardRemove()
        )
        return AMOUNT_EDIT_STATE
    else:
        await update.message.reply_text(
            "❌ Ошибка при обновлении категории. Попробуйте снова.",
            reply_markup=get_main_menu_keyboard()
        )
        return ConversationHandler.END

async def amount_edit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Шаг изменения суммы после смены категории"""
    text = update.message.text.strip().replace(',', '.')
    selected_expense = context.user_data.get('selected_expense')
    if not selected_expense:
        await update.message.reply_text(
            "Ошибка: расход не найден.",
            reply_markup=get_main_menu_keyboard()
        )
        return ConversationHandler.END

    exp_id, old_amount, desc, new_category, date = selected_expense

    # Если пользователь отправил текущее значение или пусто — не меняем сумму
    try:
        new_amount = float(text)
    except ValueError:
        new_amount = old_amount

    # Обновляем сумму, если изменилась
    if abs(float(new_amount) - float(old_amount)) > 1e-9:
        if not update_expense_amount(exp_id, new_amount):
            await update.message.reply_text(
                "❌ Не удалось обновить сумму.",
                reply_markup=get_main_menu_keyboard()
            )
            return ConversationHandler.END

    await update.message.reply_text(
        f"✅ Обновлено!\n"
        f"📝 {desc}\n"
        f"🏷️ Категория: {new_category}\n"
        f"💰 Сумма: {float(new_amount):.2f} Тг\n\n"
        f"Переобучаю модель...",
        reply_markup=get_main_menu_keyboard()
    )

    # Автоматически переобучаем модель на исправленных данных
    await retrain_model_on_corrected_data(update, context)
    return ConversationHandler.END

async def retrain_model_on_corrected_data(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Переобучение модели на исправленных данных"""
    try:
        # Получаем все расходы из базы данных
        training_data = get_all_expenses_for_training()
        
        if training_data:
            # Обучаем модель на исправленных данных
            train_model(training_data)
            
            # Также добавляем исправленные данные в словарь категорий
            for description, category in training_data:
                if category in CATEGORIES and description.lower() not in [w.lower() for w in CATEGORIES[category]]:
                    CATEGORIES[category].append(description.lower())
            
            await update.message.reply_text(
                "🤖 Модель успешно переобучена на исправленных данных!\n"
                "Теперь похожие товары будут автоматически классифицироваться правильно."
            )
        else:
            await update.message.reply_text(
                "⚠️ Нет данных для обучения модели."
            )
    except Exception as e:
        logger.error(f"Ошибка при переобучении модели: {e}")
        await update.message.reply_text(
            f"⚠️ Ошибка при переобучении модели: {e}"
        )

async def manual_training(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Ручное обучение модели на всех данных из БД"""
    user_id = update.effective_user.id
    
    # Проверяем защиту блока обучения
    if not validate_block_access("training", user_id):
        await update.message.reply_text(
            "❌ Доступ к обучению модели ограничен.",
            reply_markup=get_main_menu_keyboard()
        )
        return
    
    try:
        training_data = get_all_expenses_for_training()
        
        if training_data:
            # Обучаем модель
            train_model(training_data)
            
            # Обновляем словарь категорий
            for description, category in training_data:
                if category in CATEGORIES and description.lower() not in [w.lower() for w in CATEGORIES[category]]:
                    CATEGORIES[category].append(description.lower())
            
            await update.message.reply_text(
                f"🤖 Модель успешно обучена на {len(training_data)} записях!\n"
                "Теперь классификация будет более точной.",
                reply_markup=get_main_menu_keyboard()
            )
        else:
            await update.message.reply_text(
                "⚠️ Нет данных для обучения модели. Сначала добавьте несколько расходов.",
                reply_markup=get_main_menu_keyboard()
            )
    except Exception as e:
        logger.error(f"Ошибка при обучении модели: {e}")
        await update.message.reply_text(
            f"❌ Ошибка при обучении модели: {e}",
            reply_markup=get_main_menu_keyboard()
        )

# --- Функции для работы с напоминаниями ---
async def reminder_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Главное меню напоминаний"""
    user_id = update.effective_user.id
    
    # Проверяем защиту блока напоминаний
    if not validate_block_access("reminders", user_id):
        await update.message.reply_text(
            "❌ Доступ к напоминаниям ограничен.",
            reply_markup=get_main_menu_keyboard()
        )
        return ConversationHandler.END
    
    text = update.message.text
    
    # Если это первое нажатие на кнопку "⏰ Напоминания"
    if text == "⏰ Напоминания":
        await update.message.reply_text(
            "Выберите действие для напоминаний:",
            reply_markup=ReplyKeyboardMarkup([
                ["📝 Добавить напоминание", "📋 Список напоминаний"], 
                ["🗑️ Удалить напоминание", "🔙 Назад"]
            ], resize_keyboard=True)
        )
        return REMINDER_MENU_STATE
    
    # Обработка выбора в меню напоминаний
    elif text == "📝 Добавить напоминание":
        await update.message.reply_text(
            "Введите название напоминания:",
            reply_markup=ReplyKeyboardRemove()
        )
        return REMINDER_TITLE_STATE
    
    elif text == "📋 Список напоминаний":
        reminders = get_all_active_reminders()
        if not reminders:
            await update.message.reply_text(
                "У вас нет активных напоминаний.",
                reply_markup=get_main_menu_keyboard()
            )
            return ConversationHandler.END
        
        reminders_text = "📋 Ваши активные напоминания:\n\n"
        total_amount = 0
        
        for i, (rem_id, title, desc, amount, start_date, end_date, sent_10, sent_3, created) in enumerate(reminders, 1):
            days_left = (end_date - datetime.now().date()).days
            status = "🟢 Активно" if days_left > 0 else "🔴 Истекло"
            
            reminders_text += f"{i}. {title}\n"
            if desc:
                reminders_text += f"   📝 {desc}\n"
            reminders_text += f"   💰 {amount:.2f} Тг\n"
            reminders_text += f"   📅 {start_date.strftime('%d.%m.%Y')} - {end_date.strftime('%d.%m.%Y')}\n"
            reminders_text += f"   {status} (осталось {days_left} дней)\n\n"
            
            total_amount += amount
        
        reminders_text += f"💰 Общая сумма к оплате: {total_amount:.2f} Тг"
        
        # Добавляем кнопки управления
        keyboard = [
            ["✏️ Редактировать напоминание", "🗑️ Удалить напоминание"],
            ["🔙 Назад"]
        ]
        
        await update.message.reply_text(
            reminders_text,
            reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
        )
        
        # Сохраняем список напоминаний в контексте для последующего выбора
        context.user_data['reminders_list'] = reminders
        return REMINDER_MENU_STATE
    
    elif text == "✏️ Редактировать напоминание":
        reminders = context.user_data.get('reminders_list', [])
        if not reminders:
            await update.message.reply_text(
                "Список напоминаний не найден. Попробуйте сначала открыть список напоминаний.",
                reply_markup=get_main_menu_keyboard()
            )
            return ConversationHandler.END
        
        # Показываем список для выбора
        reminders_text = "✏️ Выберите напоминание для редактирования:\n\n"
        keyboard = []
        
        for i, (rem_id, title, desc, amount, start_date, end_date, sent_10, sent_3, created) in enumerate(reminders, 1):
            days_left = (end_date - datetime.now().date()).days
            status = "🟢" if days_left > 0 else "🔴"
            
            reminders_text += f"{i}. {status} {title} - {amount:.2f} Тг\n"
            keyboard.append([KeyboardButton(f"✏️ {i}. {title}")])
        
        keyboard.append([KeyboardButton("🔙 Назад")])
        
        await update.message.reply_text(
            reminders_text,
            reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
        )
        context.user_data['current_state'] = 'reminder_edit_choice'
        return REMINDER_EDIT_CHOICE_STATE
    
    elif text == "🗑️ Удалить напоминание":
        reminders = get_all_active_reminders()
        if not reminders:
            await update.message.reply_text(
                "У вас нет активных напоминаний для удаления.",
                reply_markup=get_main_menu_keyboard()
            )
            return ConversationHandler.END
        
        reminders_text = "🗑️ Выберите напоминание для удаления:\n\n"
        keyboard = []
        
        for i, (rem_id, title, desc, amount, start_date, end_date, sent_10, sent_3, created) in enumerate(reminders, 1):
            days_left = (end_date - datetime.now().date()).days
            reminders_text += f"{i}. {title} - {amount:.2f} Тг (осталось {days_left} дней)\n"
            keyboard.append([KeyboardButton(f"❌ Удалить {i}")])
        
        keyboard.append([KeyboardButton("🔙 Назад")])
        
        await update.message.reply_text(
            reminders_text,
            reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
        )
        context.user_data['reminders_list'] = reminders
        return REMINDER_DELETE_STATE
    
    elif text == "🔙 Назад":
        await update.message.reply_text(
            "Возвращаюсь в главное меню:",
            reply_markup=get_main_menu_keyboard()
        )
        return ConversationHandler.END
    
    return REMINDER_MENU_STATE

async def reminder_edit_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора напоминания для редактирования"""
    text = update.message.text
    
    if text == "🔙 Назад":
        context.user_data.pop('current_state', None)
        await update.message.reply_text(
            "Выберите действие для напоминаний:",
            reply_markup=ReplyKeyboardMarkup([
                ["📝 Добавить напоминание", "📋 Список напоминаний"], 
                ["🗑️ Удалить напоминание", "🔙 Назад"]
            ], resize_keyboard=True)
        )
        return REMINDER_MENU_STATE
    
    # Парсим выбор пользователя
    if text.startswith("✏️ "):
        try:
            # Извлекаем номер из текста "✏️ 1. Название"
            choice_num = int(text.split(".")[0].split()[-1]) - 1
            reminders = context.user_data.get('reminders_list', [])
            
            if 0 <= choice_num < len(reminders):
                selected_reminder = reminders[choice_num]
                context.user_data['editing_reminder'] = selected_reminder
                
                rem_id, title, desc, amount, start_date, end_date, sent_10, sent_3, created = selected_reminder
                
                await update.message.reply_text(
                    f"✏️ Редактирование напоминания:\n\n"
                    f"📝 Текущее название: {title}\n"
                    f"📄 Описание: {desc or 'Не указано'}\n"
                    f"💰 Сумма: {amount:.2f} Тг\n"
                    f"📅 Период: {start_date.strftime('%d.%m.%Y')} - {end_date.strftime('%d.%m.%Y')}\n\n"
                    f"Введите новое название (или отправьте текущее без изменений):",
                    reply_markup=ReplyKeyboardRemove()
                )
                return REMINDER_EDIT_TITLE_STATE
            else:
                await update.message.reply_text(
                    "❌ Неверный выбор. Попробуйте снова.",
                    reply_markup=ReplyKeyboardMarkup([["🔙 Назад"]], resize_keyboard=True)
                )
                return REMINDER_EDIT_CHOICE_STATE
        except (ValueError, IndexError):
            await update.message.reply_text(
                "❌ Неверный формат. Попробуйте снова.",
                reply_markup=ReplyKeyboardMarkup([["🔙 Назад"]], resize_keyboard=True)
            )
            return REMINDER_EDIT_CHOICE_STATE
    
    # Обрабатываем выбор по номеру (например "1. Автострахование")
    elif text and text[0].isdigit() and "." in text:
        try:
            # Извлекаем номер из текста "1. Название"
            choice_num = int(text.split(".")[0]) - 1
            reminders = context.user_data.get('reminders_list', [])
            
            if 0 <= choice_num < len(reminders):
                selected_reminder = reminders[choice_num]
                context.user_data['editing_reminder'] = selected_reminder
                
                rem_id, title, desc, amount, start_date, end_date, sent_10, sent_3, created = selected_reminder
                
                await update.message.reply_text(
                    f"✏️ Редактирование напоминания:\n\n"
                    f"📝 Текущее название: {title}\n"
                    f"📄 Описание: {desc or 'Не указано'}\n"
                    f"💰 Сумма: {amount:.2f} Тг\n"
                    f"📅 Период: {start_date.strftime('%d.%m.%Y')} - {end_date.strftime('%d.%m.%Y')}\n\n"
                    f"Введите новое название (или отправьте текущее без изменений):",
                    reply_markup=ReplyKeyboardRemove()
                )
                return REMINDER_EDIT_TITLE_STATE
            else:
                await update.message.reply_text(
                    "❌ Неверный выбор. Попробуйте снова.",
                    reply_markup=ReplyKeyboardMarkup([["🔙 Назад"]], resize_keyboard=True)
                )
                return REMINDER_EDIT_CHOICE_STATE
        except (ValueError, IndexError):
            await update.message.reply_text(
                "❌ Неверный формат. Попробуйте снова.",
                reply_markup=ReplyKeyboardMarkup([["🔙 Назад"]], resize_keyboard=True)
            )
            return REMINDER_EDIT_CHOICE_STATE
    
    return REMINDER_EDIT_CHOICE_STATE

async def reminder_edit_title(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Редактирование названия напоминания"""
    new_title = update.message.text.strip()
    if not new_title:
        await update.message.reply_text(
            "Название не может быть пустым. Введите новое название:",
            reply_markup=ReplyKeyboardRemove()
        )
        return REMINDER_EDIT_TITLE_STATE
    
    context.user_data['new_title'] = new_title
    
    selected_reminder = context.user_data.get('editing_reminder')
    rem_id, title, desc, amount, start_date, end_date, sent_10, sent_3, created = selected_reminder
    
    await update.message.reply_text(
        f"📝 Новое название: {new_title}\n\n"
        f"Введите новое описание (или '-' для удаления, или отправьте текущее без изменений):",
        reply_markup=ReplyKeyboardRemove()
    )
    return REMINDER_EDIT_DESC_STATE

async def reminder_edit_desc(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Редактирование описания напоминания"""
    new_desc = update.message.text.strip()
    if new_desc == '-':
        new_desc = None
    
    context.user_data['new_desc'] = new_desc
    
    selected_reminder = context.user_data.get('editing_reminder')
    rem_id, title, desc, amount, start_date, end_date, sent_10, sent_3, created = selected_reminder
    
    await update.message.reply_text(
        f"📄 Новое описание: {new_desc or 'Не указано'}\n\n"
        f"Введите новую сумму (или отправьте текущую без изменений):",
        reply_markup=ReplyKeyboardRemove()
    )
    return REMINDER_EDIT_AMOUNT_STATE

async def reminder_edit_amount(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Редактирование суммы напоминания"""
    try:
        new_amount = float(update.message.text.replace(',', '.'))
        if new_amount <= 0:
            raise ValueError
    except ValueError:
        await update.message.reply_text(
            "❌ Неверная сумма. Введите положительное число:",
            reply_markup=ReplyKeyboardRemove()
        )
        return REMINDER_EDIT_AMOUNT_STATE
    
    context.user_data['new_amount'] = new_amount
    
    selected_reminder = context.user_data.get('editing_reminder')
    rem_id, title, desc, amount, start_date, end_date, sent_10, sent_3, created = selected_reminder
    
    await update.message.reply_text(
        f"💰 Новая сумма: {new_amount:.2f} Тг\n\n"
        f"Введите новые даты в формате ДД.ММ.ГГГГ - ДД.ММ.ГГГГ\n"
        f"Текущий период: {start_date.strftime('%d.%m.%Y')} - {end_date.strftime('%d.%m.%Y')}\n"
        f"Или отправьте текущий период без изменений:",
        reply_markup=ReplyKeyboardRemove()
    )
    return REMINDER_EDIT_DATES_STATE

async def reminder_edit_dates(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Редактирование дат напоминания"""
    dates_text = update.message.text.strip()
    
    try:
        if ' - ' in dates_text:
            start_str, end_str = dates_text.split(' - ')
            new_start_date = datetime.strptime(start_str.strip(), '%d.%m.%Y').date()
            new_end_date = datetime.strptime(end_str.strip(), '%d.%m.%Y').date()
        else:
            # Если пользователь отправил текущие даты без изменений
            selected_reminder = context.user_data.get('editing_reminder')
            rem_id, title, desc, amount, start_date, end_date, sent_10, sent_3, created = selected_reminder
            new_start_date = start_date
            new_end_date = end_date
        
        if new_start_date >= new_end_date:
            raise ValueError("Дата начала должна быть раньше даты окончания")
            
    except ValueError as e:
        await update.message.reply_text(
            f"❌ Неверный формат дат. {str(e)}\n"
            f"Введите даты в формате ДД.ММ.ГГГГ - ДД.ММ.ГГГГ:",
            reply_markup=ReplyKeyboardRemove()
        )
        return REMINDER_EDIT_DATES_STATE
    
    # Сохраняем изменения в базу данных
    selected_reminder = context.user_data.get('editing_reminder')
    rem_id, title, desc, amount, start_date, end_date, sent_10, sent_3, created = selected_reminder
    
    new_title = context.user_data.get('new_title', title)
    new_desc = context.user_data.get('new_desc', desc)
    new_amount = context.user_data.get('new_amount', amount)
    
    # Обновляем напоминание в базе данных
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE payment_reminders 
                SET title = %s, description = %s, amount = %s, 
                    start_date = %s, end_date = %s, sent_10_days = FALSE, sent_3_days = FALSE
                WHERE id = %s
            ''', (new_title, new_desc, new_amount, new_start_date, new_end_date, rem_id))
            conn.commit()
            
            await update.message.reply_text(
                f"✅ Напоминание успешно обновлено!\n\n"
                f"📝 Название: {new_title}\n"
                f"📄 Описание: {new_desc or 'Не указано'}\n"
                f"💰 Сумма: {new_amount:.2f} Тг\n"
                f"📅 Период: {new_start_date.strftime('%d.%m.%Y')} - {new_end_date.strftime('%d.%m.%Y')}",
                reply_markup=get_main_menu_keyboard()
            )
        except Exception as e:
            logger.error(f"Ошибка при обновлении напоминания: {e}")
            await update.message.reply_text(
                "❌ Ошибка при обновлении напоминания. Попробуйте снова.",
                reply_markup=get_main_menu_keyboard()
            )
        finally:
            conn.close()
    else:
        await update.message.reply_text(
            "❌ Ошибка подключения к базе данных.",
            reply_markup=get_main_menu_keyboard()
        )
    
    # Очищаем контекст
    context.user_data.pop('editing_reminder', None)
    context.user_data.pop('new_title', None)
    context.user_data.pop('new_desc', None)
    context.user_data.pop('new_amount', None)
    context.user_data.pop('current_state', None)
    
    return ConversationHandler.END

async def reminder_title_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ввода названия напоминания"""
    title = update.message.text.strip()
    if not title:
        await update.message.reply_text(
            "Название не может быть пустым. Попробуйте снова:",
            reply_markup=get_main_menu_keyboard()
        )
        return ConversationHandler.END
    
    context.user_data['reminder_title'] = title
    
    await update.message.reply_text(
        f"Название: {title}\n\n"
        "Теперь введите описание (или отправьте '-' если описание не нужно):",
        reply_markup=ReplyKeyboardRemove()
    )
    return REMINDER_DESC_STATE

async def reminder_desc_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ввода описания напоминания"""
    desc = update.message.text.strip()
    if desc == '-':
        desc = None
    
    context.user_data['reminder_desc'] = desc
    
    await update.message.reply_text(
        f"Название: {context.user_data['reminder_title']}\n"
        f"Описание: {desc or 'Не указано'}\n\n"
        "Теперь введите сумму (например: 25000):",
        reply_markup=ReplyKeyboardRemove()
    )
    return REMINDER_AMOUNT_STATE

async def reminder_amount_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ввода суммы напоминания"""
    try:
        amount = float(update.message.text.replace(',', '.'))
        if amount <= 0:
            raise ValueError("Сумма должна быть положительной")
    except ValueError:
        await update.message.reply_text(
            "Неверный формат суммы. Введите число больше 0:",
            reply_markup=get_main_menu_keyboard()
        )
        return ConversationHandler.END
    
    context.user_data['reminder_amount'] = amount
    
    await update.message.reply_text(
        f"Название: {context.user_data['reminder_title']}\n"
        f"Описание: {context.user_data['reminder_desc'] or 'Не указано'}\n"
        f"Сумма: {amount:.2f} Тг\n\n"
        "Теперь введите дату начала в формате ДД.ММ.ГГГГ (например: 20.08.2025):",
        reply_markup=ReplyKeyboardRemove()
    )
    return REMINDER_START_DATE_STATE

async def reminder_start_date_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ввода даты начала"""
    try:
        start_date = datetime.strptime(update.message.text, '%d.%m.%Y').date()
    except ValueError:
        await update.message.reply_text(
            "Неверный формат даты. Используйте формат ДД.ММ.ГГГГ:",
            reply_markup=get_main_menu_keyboard()
        )
        return ConversationHandler.END
    
    context.user_data['reminder_start_date'] = start_date
    
    await update.message.reply_text(
        f"Название: {context.user_data['reminder_title']}\n"
        f"Описание: {context.user_data['reminder_desc'] or 'Не указано'}\n"
        f"Сумма: {context.user_data['reminder_amount']:.2f} Тг\n"
        f"Дата начала: {start_date.strftime('%d.%m.%Y')}\n\n"
        "Теперь введите дату окончания в формате ДД.ММ.ГГГГ (например: 19.08.2026):",
        reply_markup=ReplyKeyboardRemove()
    )
    return REMINDER_END_DATE_STATE

async def reminder_end_date_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ввода даты окончания"""
    try:
        end_date = datetime.strptime(update.message.text, '%d.%m.%Y').date()
        start_date = context.user_data['reminder_start_date']
        
        if end_date <= start_date:
            await update.message.reply_text(
                "Дата окончания должна быть позже даты начала. Попробуйте снова:",
                reply_markup=get_main_menu_keyboard()
            )
            return ConversationHandler.END
            
    except ValueError:
        await update.message.reply_text(
            "Неверный формат даты. Используйте формат ДД.ММ.ГГГГ:",
            reply_markup=get_main_menu_keyboard()
        )
        return ConversationHandler.END
    
    # Сохраняем напоминание в базу данных
    title = context.user_data['reminder_title']
    desc = context.user_data['reminder_desc']
    amount = context.user_data['reminder_amount']
    start_date = context.user_data['reminder_start_date']
    
    if add_payment_reminder(title, desc, amount, start_date, end_date):
        days_left = (end_date - datetime.now().date()).days
        
        await update.message.reply_text(
            f"✅ Напоминание успешно добавлено!\n\n"
            f"📋 {title}\n"
            f"📝 {desc or 'Описание не указано'}\n"
            f"💰 {amount:.2f} Тг\n"
            f"📅 {start_date.strftime('%d.%m.%Y')} - {end_date.strftime('%d.%m.%Y')}\n"
            f"⏰ Осталось дней: {days_left}\n\n"
            f"Бот будет напоминать о необходимости оплаты за 10 и 3 дня до истечения срока.",
            reply_markup=get_main_menu_keyboard()
        )
    else:
        await update.message.reply_text(
            "❌ Ошибка при сохранении напоминания. Попробуйте снова.",
            reply_markup=get_main_menu_keyboard()
        )
    
    # Очищаем данные
    context.user_data.clear()
    return ConversationHandler.END

async def reminder_manage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Управление напоминаниями (удаление)"""
    text = update.message.text
    
    if text == "🔙 Назад":
        await update.message.reply_text(
            "Возвращаюсь в меню напоминаний:",
            reply_markup=ReplyKeyboardMarkup([
                ["📝 Добавить напоминание", "📋 Список напоминаний"], 
                ["🔙 Назад"]
            ], resize_keyboard=True)
        )
        return REMINDER_MENU_STATE
    
    # Проверяем формат "❌ Удалить N"
    if text.startswith("❌ Удалить "):
        try:
            reminder_num = int(text.split()[-1]) - 1
            reminders = context.user_data.get('reminders_list', [])
            
            if 0 <= reminder_num < len(reminders):
                reminder = reminders[reminder_num]
                reminder_id = reminder[0]
                reminder_title = reminder[1]
                
                if delete_reminder(reminder_id):
                    await update.message.reply_text(
                        f"✅ Напоминание '{reminder_title}' успешно удалено!",
                        reply_markup=get_main_menu_keyboard()
                    )
                    return ConversationHandler.END
                else:
                    await update.message.reply_text(
                        "❌ Ошибка при удалении напоминания. Попробуйте снова.",
                        reply_markup=get_main_menu_keyboard()
                    )
            else:
                await update.message.reply_text(
                    "Неверный номер напоминания.",
                    reply_markup=get_main_menu_keyboard()
                )
                return ConversationHandler.END
                
        except (ValueError, IndexError):
            await update.message.reply_text(
                "Неверный формат команды.",
                reply_markup=get_main_menu_keyboard()
            )
            return ConversationHandler.END
    
    return ConversationHandler.END

async def period_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    period_text = update.message.text.lower()
    start_date, end_date = parse_date_period(period_text)
    if not start_date:
        await update.message.reply_text("Не могу распознать период.", reply_markup=get_main_menu_keyboard())
        return ConversationHandler.END

    conn = get_db_connection()
    if not conn:
        await update.message.reply_text("Проблема с подключением к базе данных.", reply_markup=get_main_menu_keyboard())
        return ConversationHandler.END

    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT description, category, amount, transaction_date
            FROM expenses
            WHERE transaction_date BETWEEN %s AND %s
            ORDER BY transaction_date ASC
        ''', (start_date, end_date))
        data = cursor.fetchall()
    except Exception as e:
        await update.message.reply_text(f"Произошла ошибка при получении отчета: {e}", reply_markup=get_main_menu_keyboard())
        return ConversationHandler.END
    finally:
        conn.close()

    if not data:
        await update.message.reply_text("За выбранный период нет расходов.", reply_markup=get_main_menu_keyboard())
        return ConversationHandler.END

    # Создание DataFrame с полными данными
    try:
        df = pd.DataFrame(data, columns=['Описание', 'Категория', 'Сумма', 'Дата транзакции'])
        df['Месяц'] = pd.to_datetime(df['Дата транзакции']).dt.strftime('%b')
        df['День недели'] = pd.to_datetime(df['Дата транзакции']).dt.strftime('%a')
        df['Неделя'] = pd.to_datetime(df['Дата транзакции']).dt.isocalendar().week
        
        # Группировка данных для различных графиков
        grouped_by_category = df.groupby('Категория', as_index=False)['Сумма'].sum().sort_values(by='Сумма', ascending=False)
        grouped_by_month = df.groupby('Месяц', as_index=False)['Сумма'].sum()
        grouped_by_week = df.groupby('Неделя', as_index=False)['Сумма'].sum()
        
        categories = grouped_by_category['Категория'].tolist()
        amounts = grouped_by_category['Сумма'].tolist()
        total = df['Сумма'].sum()
        
        # Статистика
        avg_expense = df['Сумма'].mean()
        max_expense = df['Сумма'].max()
        min_expense = df['Сумма'].min()
        total_transactions = len(df)
        
    except Exception as e:
        logger.error(f"Ошибка при создании DataFrame: {e}")
        return ConversationHandler.END

    # Создание Excel файла
    excel_buf = io.BytesIO()
    df.to_excel(excel_buf, index=False, engine='xlsxwriter')
    excel_buf.seek(0)

    # Создание отчета в зависимости от периода
    if 'сегодня' in period_text:
        fig = create_today_report(df, grouped_by_category, categories, amounts, total)
    elif 'неделя' in period_text:
        fig = create_week_report(df, grouped_by_category, categories, amounts, total)
    elif 'месяц' in period_text:
        fig = create_month_report(df, grouped_by_category, grouped_by_week, categories, amounts, total)
    elif 'год' in period_text:
        fig = create_year_report(df, grouped_by_category, grouped_by_month, categories, amounts, total)
    else:
        # Fallback для неизвестных периодов
        fig = create_today_report(df, grouped_by_category, categories, amounts, total)

    # Сохранение графика
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, 
                facecolor='#1a1a1a', edgecolor='none')
    buf.seek(0)
    plt.close(fig)

    # Текстовая сводка
    summary_text = f"📊 ОТЧЕТ ЗА {period_text.upper()}\n\n"
    summary_text += f"💰 Общие расходы: {total:.2f} Тг\n"
    summary_text += f"📈 Средний расход: {avg_expense:.2f} Тг\n"
    summary_text += f"🔄 Транзакций: {total_transactions}\n"
    summary_text += f"🏷️ Категорий: {len(categories)}\n\n"
    summary_text += "📋 Топ категории:\n"
    for i, (cat, amt) in enumerate(zip(categories[:5], amounts[:5]), 1):
        summary_text += f"{i}. {cat}: {amt:.2f} Тг\n"
    
    # Добавляем информацию о предстоящих платежах
    upcoming_reminders = get_upcoming_reminders(90)  # На 90 дней вперед
    if upcoming_reminders:
        summary_text += "\n⏰ ПРЕДСТОЯЩИЕ ПЛАТЕЖИ:\n"
        total_upcoming = 0
        for rem_id, title, desc, amount, start_date, end_date, sent_10, sent_3 in upcoming_reminders[:5]:  # Показываем топ-5
            days_left = (end_date - datetime.now().date()).days
            if days_left > 0:
                summary_text += f"• {title}: {amount:.2f} Тг (через {days_left} дней)\n"
                total_upcoming += amount
        if total_upcoming > 0:
            summary_text += f"💰 Общая сумма: {total_upcoming:.2f} Тг\n"
    
    # Отправка отчета и сводки
    await update.message.reply_photo(photo=buf, caption=summary_text, reply_markup=get_main_menu_keyboard())

    # Отправка Excel файла
    await update.message.reply_document(document=excel_buf, filename=f"Отчет_{period_text}.xlsx")
    return ConversationHandler.END

def create_today_report(df, grouped_by_category, categories, amounts, total):
    """Создание отчета за сегодня - красивый пирог с понятными тегами"""
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#1a1a1a')
    
    # Цветовая палитра
    colors = ['#6B8E23', '#4682B4', '#CD853F', '#20B2AA', '#8A2BE2', '#32CD32', '#FF8C00', '#DC143C', '#1E90FF', '#9370DB']
    
    # Создаем красивый пирог
    wedges, texts, autotexts = ax.pie(amounts, labels=categories, autopct='%1.1f%%', 
                                      startangle=90, colors=colors[:len(amounts)],
                                      textprops={'fontsize': 14, 'fontweight': 'bold', 'color': 'white'},
                                      shadow=True, explode=[0.05] * len(amounts))
    
    # Настройка процентов
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(13)
    
    # Центральный текст
    ax.text(0, 0, f'ОБЩИЕ\nРАСХОДЫ\n{total:.0f} Тг', ha='center', va='center', 
            fontsize=20, fontweight='bold', color='white')
    
    ax.set_title('РАСХОДЫ ЗА СЕГОДНЯ', color='white', fontsize=22, fontweight='bold', pad=30)
    
    # Легенда справа
    legend_labels = [f"{cat} — {amt:.0f} Тг" for cat, amt in zip(categories, amounts)]
    ax.legend(wedges, legend_labels, title="Категории", loc="center left", 
             bbox_to_anchor=(1.1, 0.5), fontsize=13, title_fontsize=15)
    
    return fig

def create_week_report(df, grouped_by_category, categories, amounts, total):
    """Создание отчета за неделю - только пирог категорий"""
    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor('#1a1a1a')
    
    # Цветовая палитра
    colors = ['#6B8E23', '#4682B4', '#CD853F', '#20B2AA', '#8A2BE2', '#32CD32', '#FF8C00', '#DC143C', '#1E90FF', '#9370DB']
    
    # Пирог категорий (полная ширина)
    ax1 = fig.add_subplot(1, 1, 1)
    wedges, texts, autotexts = ax1.pie(amounts, labels=None, autopct='%1.1f%%', 
                                       startangle=90, colors=colors[:len(amounts)],
                                       shadow=True, explode=[0.05] * len(amounts))
    
    # Настройка процентов
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(13)
    
    # Центральный текст
    ax1.text(0, 0, f'ОБЩИЕ\nРАСХОДЫ\n{total:.0f} Тг', ha='center', va='center', 
            fontsize=18, fontweight='bold', color='white')
    
    ax1.set_title('РАСХОДЫ ПО КАТЕГОРИЯМ', color='white', fontsize=18, fontweight='bold', pad=20)
    
    # Легенда для пирога
    legend_labels = [f"{cat} — {amt:.0f} Тг" for cat, amt in zip(categories, amounts)]
    ax1.legend(wedges, legend_labels, title="Категории", loc="center left", 
              bbox_to_anchor=(1.0, 0.5), fontsize=12, title_fontsize=14)
    
    fig.suptitle('ОТЧЕТ ЗА НЕДЕЛЮ', color='white', fontsize=22, fontweight='bold', y=0.95)
    
    return fig

def create_month_report(df, grouped_by_category, grouped_by_week, categories, amounts, total):
    """Создание отчета за месяц - пирог и сравнение недель"""
    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor('#1a1a1a')
    
    # Цветовая палитра
    colors = ['#6B8E23', '#4682B4', '#CD853F', '#20B2AA', '#8A2BE2', '#32CD32', '#FF8C00', '#DC143C', '#1E90FF', '#9370DB']
    
    # 1. Пирог категорий (левая часть)
    ax1 = fig.add_subplot(1, 2, 1)
    wedges, texts, autotexts = ax1.pie(amounts, labels=None, autopct='%1.1f%%', 
                                       startangle=90, colors=colors[:len(amounts)],
                                       shadow=True, explode=[0.05] * len(amounts))
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(13)
    
    ax1.text(0, 0, f'ОБЩИЕ\nРАСХОДЫ\n{total:.0f} Тг', ha='center', va='center', 
            fontsize=16, fontweight='bold', color='white')
    
    ax1.set_title('РАСХОДЫ ПО КАТЕГОРИЯМ', color='white', fontsize=16, fontweight='bold', pad=20)
    
    # 2. Сравнение недель (правая часть)
    ax2 = fig.add_subplot(1, 2, 2)
    weeks = grouped_by_week['Неделя'].tolist()
    week_amounts = grouped_by_week['Сумма'].tolist()
    
    bars = ax2.bar(weeks, week_amounts, color=colors[:len(weeks)], alpha=0.8)
    ax2.set_title('СРАВНЕНИЕ НЕДЕЛЬ', color='white', fontsize=16, fontweight='bold', pad=20)
    ax2.set_ylabel('Сумма (Тг)', color='white', fontsize=14)
    ax2.set_xlabel('Номер недели', color='white', fontsize=14)
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    # Добавляем значения на столбцы
    for bar, amount in zip(bars, week_amounts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(week_amounts)*0.01,
                 f'{amount:.0f}', ha='center', va='bottom', color='white', fontweight='bold', fontsize=12)
    
    # Легенда для пирога
    legend_labels = [f"{cat} — {amt:.0f} Тг" for cat, amt in zip(categories, amounts)]
    ax1.legend(wedges, legend_labels, title="Категории", loc="upper left", 
              bbox_to_anchor=(-0.1, 1.0), fontsize=11, title_fontsize=13)
    
    fig.suptitle('ОТЧЕТ ЗА МЕСЯЦ', color='white', fontsize=22, fontweight='bold', y=0.95)
    
    return fig

def create_year_report(df, grouped_by_category, grouped_by_month, categories, amounts, total):
    """Создание отчета за год - пирог и сравнение месяцев"""
    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor('#1a1a1a')
    
    # Цветовая палитра
    colors = ['#6B8E23', '#4682B4', '#CD853F', '#20B2AA', '#8A2BE2', '#32CD32', '#FF8C00', '#DC143C', '#1E90FF', '#9370DB']
    
    # 1. Пирог категорий (левая часть)
    ax1 = fig.add_subplot(1, 2, 1)
    wedges, texts, autotexts = ax1.pie(amounts, labels=None, autopct='%1.1f%%', 
                                       startangle=90, colors=colors[:len(amounts)],
                                       shadow=True, explode=[0.05] * len(amounts))
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(13)
    
    ax1.text(0, 0, f'ОБЩИЕ\nРАСХОДЫ\n{total:.0f} Тг', ha='center', va='center', 
            fontsize=16, fontweight='bold', color='white')
    
    ax1.set_title('РАСХОДЫ ПО КАТЕГОРИЯМ', color='white', fontsize=16, fontweight='bold', pad=20)
    
    # 2. Сравнение месяцев (правая часть)
    ax2 = fig.add_subplot(1, 2, 2)
    months = grouped_by_month['Месяц'].tolist()
    month_amounts = grouped_by_month['Сумма'].tolist()
    
    bars = ax2.bar(months, month_amounts, color=colors[:len(months)], alpha=0.8)
    ax2.set_title('СРАВНЕНИЕ МЕСЯЦЕВ', color='white', fontsize=16, fontweight='bold', pad=20)
    ax2.set_ylabel('Сумма (Тг)', color='white', fontsize=14)
    ax2.set_xlabel('Месяц', color='white', fontsize=14)
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    # Добавляем значения на столбцы
    for bar, amount in zip(bars, month_amounts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(month_amounts)*0.01,
                 f'{amount:.0f}', ha='center', va='bottom', color='white', fontweight='bold', fontsize=12)
    
    # Легенда для пирога
    legend_labels = [f"{cat} — {amt:.0f} Тг" for cat, amt in zip(categories, amounts)]
    ax1.legend(wedges, legend_labels, title="Категории", loc="upper left", 
              bbox_to_anchor=(-0.1, 1.0), fontsize=11, title_fontsize=13)
    
    fig.suptitle('ОТЧЕТ ЗА ГОД', color='white', fontsize=22, fontweight='bold', y=0.95)
    
    return fig

def parse_date_period(text):
    text_lower = text.lower()
    start_date = None
    end_date = datetime.now(timezone.utc)
    current_time_aware = datetime.now(timezone.utc)

    if 'сегодня' in text_lower:
        start_date = current_time_aware.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=1, microseconds=-1)
    elif 'неделя' in text_lower:
        start_date = (current_time_aware - timedelta(days=current_time_aware.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = current_time_aware.replace(hour=23, minute=59, second=59, microsecond=999999)
    elif 'месяц' in text_lower:
        start_date = current_time_aware.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_date = current_time_aware.replace(hour=23, minute=59, second=59, microsecond=999999)
    elif 'год' in text_lower:
        start_date = current_time_aware.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        end_date = current_time_aware.replace(hour=23, minute=59, second=59, microsecond=999999)
    else:
        start_date = None
    return start_date, end_date

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    
    # Проверяем, находится ли пользователь в состоянии ожидания username
    if context.user_data.get('auth_state') == 'waiting_for_username':
        # Обрабатываем ввод username
        text = update.message.text.strip()
        
        if text == "🔙 Отмена":
            await update.message.reply_text(
                "❌ Доступ к боту отменен. Обратитесь к администратору.",
                reply_markup=ReplyKeyboardRemove()
            )
            context.user_data.pop('auth_state', None)
            return
        
        # Проверяем username
        if len(text) < 2:
            await update.message.reply_text(
                "❌ Имя должно содержать минимум 2 символа.\n\n"
                "Попробуйте еще раз:",
                reply_markup=ReplyKeyboardMarkup([["🔙 Отмена"]], resize_keyboard=True)
            )
            return
        
        # Проверяем, есть ли username в списке авторизованных
        if is_username_authorized(text):
            logger.info(f"Пользователь {user_id} авторизован по имени '{text}'")
            
            # Обновляем telegram_id для этого пользователя
            users_data = load_authorized_users()
            for user in users_data.get("users", []):
                if user.get("username") == text:
                    user["telegram_id"] = user_id
                    save_authorized_users(users_data)
                    logger.info(f"Обновлен telegram_id для пользователя '{text}': {user_id}")
                    
                    # Обновляем данные в базе данных Railway
                    update_user_telegram_id(text, user_id)
                    break
            
            await update.message.reply_text(
                "✅ Ваше имя найдено в списке авторизованных пользователей!\n\n"
                "Теперь у вас есть доступ к боту!",
                reply_markup=get_main_menu_keyboard()
            )
            context.user_data.pop('auth_state', None)
            return
        else:
            await update.message.reply_text(
                "❌ Ваше имя не найдено в списке авторизованных пользователей.\n\n"
                "Обратитесь к администратору для добавления в список:",
                reply_markup=ReplyKeyboardMarkup([["🔙 Отмена"]], resize_keyboard=True)
            )
            return
    
    # Проверяем защиту блока расходов
    if not validate_block_access("expenses", user_id):
        await update.message.reply_text(
            "❌ Доступ к добавлению расходов ограничен.",
            reply_markup=get_main_menu_keyboard()
        )
        return
    
    text = update.message.text.strip()
    
    # Проверяем специальные команды
    if text == "🔧 Исправить категории":
        await correction_menu(update, context)
        return
    elif text == "📚 Обучить модель":
        await manual_training(update, context)
        return
    elif text == "⏰ Напоминания":
        await reminder_menu(update, context)
        return
    elif text == "📅 Планирование":
        await planning_menu(update, context)
        return
    elif text == "📈 Аналитика":
        await analytics_menu(update, context)
        return
    elif text == "👥 Управление группой":
        await group_management_menu(update, context)
        return
    elif text == "✏️ Редактировать напоминание":
        await reminder_menu(update, context)
        return
    elif text == "✏️ Редактировать план":
        await planning_menu(update, context)
        return
    elif text == "🗑️ Удалить напоминание":
        await reminder_menu(update, context)
        return
    elif text == "🗑️ Удалить план":
        await planning_menu(update, context)
        return
    elif text == "📋 Список напоминаний":
        await reminder_menu(update, context)
        return
    elif text == "📋 Список планов":
        await planning_menu(update, context)
        return
    # Проверяем выбор элементов для редактирования
    elif text and text.startswith("✏️ ") and "." in text:
        # Это выбор элемента для редактирования - передаем в соответствующий обработчик
        if context.user_data.get('current_state') == 'reminder_edit_choice':
            await reminder_edit_choice(update, context)
            return
        elif context.user_data.get('current_state') == 'plan_edit_choice':
            await planning_edit_choice(update, context)
            return
    elif text and text[0].isdigit() and "." in text and not any(x in text for x in ["💸", "📊", "⏰", "📅", "📈", "👥"]):
        # Это выбор по номеру (например "1. Автострахование" или "1. 09.2025")
        if context.user_data.get('current_state') == 'reminder_edit_choice':
            await reminder_edit_choice(update, context)
            return
        elif context.user_data.get('current_state') == 'plan_edit_choice':
            await planning_edit_choice(update, context)
            return
    elif text in ["💸 Добавить расход", "📊 Отчеты", "Сегодня", "Неделя", "Месяц", "Год"]:
        if text == "💸 Добавить расход":
            await update.message.reply_text(
                "💸 Для добавления расхода напишите в формате:\n\n"
                "📝 Описание Сумма\n\n"
                "Например:\n"
                "• Обед в кафе 1500\n"
                "• Такси домой 800\n"
                "• Продукты 2500\n\n"
                "Бот автоматически определит категорию и запишет расход!",
                reply_markup=get_main_menu_keyboard()
            )
        return

    logger.info(f"Получено сообщение: {text}")

    match = re.match(r"(.+?)\s+(\d+[.,]?\d*)$", text)
    if not match:
        await update.message.reply_text(
            "Неверный формат. Используйте: 'Описание Сумма' (например, 'Обед в кафе 150').",
            reply_markup=get_main_menu_keyboard()
        )
        return
    
    description = match.group(1).strip()
    amount_str = match.group(2).replace(',', '.')
    
    try:
        amount = float(amount_str)
        category = classify_expense(description)
        transaction_date = datetime.now(timezone.utc)
        if add_expense(amount, category, description, transaction_date, user_id): 
            await update.message.reply_text(
                f"✅ Расход '{description}' ({amount:.2f}) записан в категорию '{category}'!\n\n"
                f"💡 Если категория неправильная, используйте '🔧 Исправить категории' для исправления.",
                reply_markup=get_main_menu_keyboard()
            )
        else:
            await update.message.reply_text(
                "Произошла ошибка при записи расхода. Пожалуйста, попробуйте ещё раз.",
                reply_markup=get_main_menu_keyboard()
            )
    except ValueError:
        await update.message.reply_text(
            "Неверный формат суммы. Сумма должна быть числом (например, 150.50).",
            reply_markup=get_main_menu_keyboard()
        )
    except Exception as e:
        logger.error(f"Непредвиденная ошибка при обработке сообщения: {e}")
        await update.message.reply_text(f"Произошла непредвиденная ошибка: {e}", reply_markup=get_main_menu_keyboard())

# --- Главная функция запуска бота ---
PERIOD_CHOICE_STATE = 1
EXPENSE_CHOICE_STATE = 2
CATEGORY_CHOICE_STATE = 3
AMOUNT_EDIT_STATE = 4
CUSTOM_CATEGORY_STATE = 5
CORRECTION_MENU_STATE = 6
EXPENSE_DELETE_STATE = 7
EXPENSE_DELETE_CONFIRM_STATE = 8
REMINDER_MENU_STATE = 9
REMINDER_TITLE_STATE = 10
REMINDER_DESC_STATE = 11
REMINDER_AMOUNT_STATE = 12
REMINDER_START_DATE_STATE = 13
REMINDER_END_DATE_STATE = 14
REMINDER_MANAGE_STATE = 15
REMINDER_DELETE_STATE = 16
REMINDER_EDIT_CHOICE_STATE = 17
REMINDER_EDIT_TITLE_STATE = 18
REMINDER_EDIT_DESC_STATE = 19
REMINDER_EDIT_AMOUNT_STATE = 20
REMINDER_EDIT_DATES_STATE = 21

# --- Доп. состояния для планирования бюджета ---
PLAN_MENU_STATE = 22
PLAN_MONTH_STATE = 23
PLAN_TOTAL_STATE = 24
PLAN_CATEGORY_STATE = 25
PLAN_AMOUNT_STATE = 26
PLAN_COMMENT_STATE = 27
PLAN_SUMMARY_STATE = 28
PLAN_DELETE_STATE = 29
PLAN_EDIT_CHOICE_STATE = 33
PLAN_EDIT_MONTH_STATE = 34
PLAN_EDIT_TOTAL_STATE = 35
PLAN_EDIT_CATEGORY_STATE = 36
PLAN_EDIT_AMOUNT_STATE = 37
PLAN_EDIT_COMMENT_STATE = 38
PLAN_EDIT_DETAILS_STATE = 39
PLAN_EDIT_CATEGORY_CHOICE_STATE = 40

# --- Состояния для аналитики ---
ANALYTICS_MENU_STATE = 30
ANALYTICS_MONTH_STATE = 31
ANALYTICS_REPORT_STATE = 32

# --- Функции для аналитики ---
async def analytics_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать меню аналитики"""
    user_id = update.effective_user.id
    
    # Проверяем защиту блока аналитики
    if not validate_block_access("analytics", user_id):
        await update.message.reply_text(
            "❌ Доступ к аналитике ограничен.",
            reply_markup=get_main_menu_keyboard()
        )
        return ConversationHandler.END
    
    keyboard = [
        [KeyboardButton("📊 Сравнение с планом")],
        [KeyboardButton("🔙 Назад")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)
    await update.message.reply_text(
        "📊 Выберите действие:",
        reply_markup=reply_markup
    )
    return ANALYTICS_MENU_STATE

async def analytics_month_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка выбора в меню аналитики"""
    text = update.message.text
    
    if text == "📊 Сравнение с планом":
        # Получаем доступные месяцы для анализа
        months = get_available_months_for_analytics()
        
        if not months:
            await update.message.reply_text(
                "❌ Нет данных для анализа. Сначала создайте план бюджета и добавьте расходы.",
                reply_markup=get_main_menu_keyboard()
            )
            return ConversationHandler.END
        
        # Формируем список месяцев
        keyboard = []
        for month_text in months:
            keyboard.append([KeyboardButton(month_text)])
        
        keyboard.append([KeyboardButton("🔙 Назад")])
        reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)
        
        await update.message.reply_text(
            "📅 Выберите месяц для сравнения:",
            reply_markup=reply_markup
        )
        return ANALYTICS_MONTH_STATE
    
    elif text == "🔙 Назад":
        await update.message.reply_text(
            "Главное меню:",
            reply_markup=get_main_menu_keyboard()
        )
        return ConversationHandler.END
    
    else:
        await update.message.reply_text(
            "❌ Неверный выбор. Используйте кнопки меню.",
            reply_markup=get_main_menu_keyboard()
        )
        return ConversationHandler.END

async def analytics_month_selected(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка выбора месяца для анализа"""
    text = update.message.text.strip()
    
    if text == "🔙 Назад":
        await analytics_menu(update, context)
        return ANALYTICS_MENU_STATE
    
    # Парсим месяц и год
    month, year = parse_month_year(text)
    if not month or not year:
        await update.message.reply_text(
            "❌ Неверный формат месяца. Используйте формат: 'Август 2025'",
            reply_markup=get_main_menu_keyboard()
        )
        return ConversationHandler.END
    
    # Генерируем простое сравнение
    await generate_simple_comparison(update, context, month, year)
    return ConversationHandler.END

def get_available_months_for_analytics():
    """Получить доступные месяцы для анализа (где есть и план и расходы)"""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor()
        # Получаем месяцы, где есть и планирование и расходы
        cursor.execute('''
            SELECT DISTINCT 
                EXTRACT(MONTH FROM bp.plan_month) as month,
                EXTRACT(YEAR FROM bp.plan_month) as year
            FROM budget_plans bp
            WHERE EXISTS (
                SELECT 1 FROM expenses e 
                WHERE EXTRACT(MONTH FROM e.transaction_date) = EXTRACT(MONTH FROM bp.plan_month)
                AND EXTRACT(YEAR FROM e.transaction_date) = EXTRACT(YEAR FROM bp.plan_month)
            )
            ORDER BY year DESC, month DESC
        ''')
        
        months = []
        for row in cursor.fetchall():
            month, year = int(row[0]), int(row[1])
            month_name = get_month_name(month)
            months.append(f"{month_name} {year}")
        
        return months
    except Exception as e:
        logger.error(f"Ошибка при получении месяцев для аналитики: {e}")
        return []
    finally:
        conn.close()

def get_month_name(month):
    """Получить название месяца по номеру"""
    months = {
        1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель",
        5: "Май", 6: "Июнь", 7: "Июль", 8: "Август",
        9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь"
    }
    return months.get(month, f"Месяц {month}")

def parse_month_year(month_text):
    """Парсинг месяца и года из текста"""
    month_names = {
        "Январь": 1, "Февраль": 2, "Март": 3, "Апрель": 4,
        "Май": 5, "Июнь": 6, "Июль": 7, "Август": 8,
        "Сентябрь": 9, "Октябрь": 10, "Ноябрь": 11, "Декабрь": 12
    }
    
    for name, num in month_names.items():
        if name in month_text:
            year = int(month_text.split()[-1])
            return num, year
    
    raise ValueError("Неверный формат месяца")

async def generate_simple_comparison(update: Update, context: ContextTypes.DEFAULT_TYPE, month: int, year: int):
    """Простое сравнение планов и расходов по категориям с графиком и Excel файлом"""
    try:
        # Получаем план бюджета
        plan_data = get_budget_plan_by_month(month, year)
        if not plan_data:
            await update.message.reply_text(
                f"❌ План бюджета на {get_month_name(month)} {year} не найден.",
                reply_markup=get_main_menu_keyboard()
            )
            return
        
        plan_id, total_budget = plan_data
        
        # Получаем детали плана по категориям
        plan_items = get_budget_plan_items(plan_id)
        plan_dict = {item[0]: item[1] for item in plan_items}  # категория: сумма
        
        # Получаем расходы за месяц
        expenses = get_monthly_expenses(month, year)
        expense_dict = {item[0]: item[1] for item in expenses}  # категория: сумма
        
        # Получаем все уникальные категории
        all_categories = set(plan_dict.keys()) | set(expense_dict.keys())
        
        # Формируем отчет
        report = f"📊 Сравнение планов и расходов за {get_month_name(month)} {year}\n\n"
        report += f"💰 Общий бюджет: {total_budget:,.0f} ₸\n\n"
        
        total_planned = sum(plan_dict.values())
        total_spent = sum(expense_dict.values())
        
        report += f"📋 Планируемые расходы: {total_planned:,.0f} ₸\n"
        report += f"💸 Фактические расходы: {total_spent:,.0f} ₸\n"
        
        if total_spent > total_planned:
            report += f"⚠️ Превышение: {total_spent - total_planned:,.0f} ₸\n"
        else:
            report += f"✅ Экономия: {total_planned - total_spent:,.0f} ₸\n"
        
        report += "\n📈 Детали по категориям:\n"
        report += "─" * 50 + "\n"
        
        for category in sorted(all_categories):
            planned = plan_dict.get(category, 0)
            spent = expense_dict.get(category, 0)
            
            report += f"\n🔸 {category}:\n"
            report += f"   План: {planned:,.0f} ₸\n"
            report += f"   Факт: {spent:,.0f} ₸\n"
            
            if spent > planned:
                report += f"   ⚠️ Превышение: {spent - planned:,.0f} ₸\n"
            elif spent < planned:
                report += f"   ✅ Экономия: {planned - spent:,.0f} ₸\n"
            else:
                report += f"   ✅ В рамках плана\n"
        
        # Создаем круговой график
        await create_analytics_chart(update, context, month, year, plan_dict, expense_dict, all_categories)
        
        # Отправляем отчет
        await update.message.reply_text(
            report,
            reply_markup=get_main_menu_keyboard()
        )
        
    except Exception as e:
        await update.message.reply_text(
            f"❌ Ошибка при генерации отчета: {str(e)}",
            reply_markup=get_main_menu_keyboard()
        )

async def create_analytics_chart(update: Update, context: ContextTypes.DEFAULT_TYPE, month: int, year: int, plan_dict: dict, expense_dict: dict, all_categories: set):
    """Создание кругового графика для аналитики"""
    try:
        # Создаем фигуру с двумя подграфиками
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Подготовка данных для планов
        plan_labels = []
        plan_sizes = []
        plan_colors = []
        
        for category in sorted(all_categories):
            if category in plan_dict and plan_dict[category] > 0:
                plan_labels.append(category)
                plan_sizes.append(plan_dict[category])
                plan_colors.append(plt.cm.Set3(len(plan_labels) % 12))
        
        # Подготовка данных для расходов
        expense_labels = []
        expense_sizes = []
        expense_colors = []
        
        for category in sorted(all_categories):
            if category in expense_dict and expense_dict[category] > 0:
                expense_labels.append(category)
                expense_sizes.append(expense_dict[category])
                expense_colors.append(plt.cm.Set3(len(expense_labels) % 12))
        
        # График планов
        if plan_sizes:
            wedges1, texts1, autotexts1 = ax1.pie(plan_sizes, labels=plan_labels, autopct='%1.1f%%', 
                                                   colors=plan_colors, startangle=90, shadow=True, explode=[0.05] * len(plan_sizes))
            ax1.set_title(f'📋 Планы на {get_month_name(month)} {year}', fontsize=16, fontweight='bold', pad=20)
            
            # Настройка текста
            for autotext in autotexts1:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(10)
        
        # График расходов
        if expense_sizes:
            wedges2, texts2, autotexts2 = ax2.pie(expense_sizes, labels=expense_labels, autopct='%1.1f%%', 
                                                  colors=expense_colors, startangle=90, shadow=True, explode=[0.05] * len(expense_sizes))
            ax2.set_title(f'💸 Фактические расходы за {get_month_name(month)} {year}', fontsize=16, fontweight='bold', pad=20)
            
            # Настройка текста
            for autotext in autotexts2:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(10)
        
        plt.tight_layout()
        
        # Сохраняем график в байты
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        img_buffer.seek(0)
        plt.close()
        
        # Отправляем график
        await update.message.reply_photo(
            photo=img_buffer,
            caption=f"📊 График сравнения планов и расходов за {get_month_name(month)} {year}",
            reply_markup=get_main_menu_keyboard()
        )
        
    except Exception as e:
        logger.error(f"Ошибка при создании графика аналитики: {e}")
        await update.message.reply_text(
            f"❌ Ошибка при создании графика: {str(e)}",
            reply_markup=get_main_menu_keyboard()
        )


        


def main():
    train_model(TRAINING_DATA)
    init_db()
    application = Application.builder().token(BOT_TOKEN).build()

    # Обработчик для отчетов
    report_conv_handler = ConversationHandler(
        entry_points=[
            MessageHandler(filters.Regex("^📊 Отчеты$"), report_menu),
            CommandHandler("report", report_menu)
        ],
        states={
            PERIOD_CHOICE_STATE: [MessageHandler(filters.Regex("^(Сегодня|Неделя|Месяц|Год)$"), period_choice)],
        },
        fallbacks=[CommandHandler("start", start)],
        allow_reentry=True
    )
    
    # Обработчик для исправления категорий
    correction_conv_handler = ConversationHandler(
        entry_points=[
            MessageHandler(filters.Regex("^🔧 Исправить категории$"), correction_menu),
            CommandHandler("correct", correction_menu)
        ],
        states={
            CORRECTION_MENU_STATE: [MessageHandler(filters.Regex("^(1️⃣ Исправить расход|2️⃣ Удалить расход|3️⃣ Назад)$"), correction_menu_choice)],
            EXPENSE_CHOICE_STATE: [MessageHandler(filters.Regex("^[0-9]+$"), expense_choice)],
            CATEGORY_CHOICE_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, category_choice)],
            CUSTOM_CATEGORY_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, custom_category_input)],
            AMOUNT_EDIT_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, amount_edit)],
            EXPENSE_DELETE_STATE: [MessageHandler(filters.Regex("^[0-9]+$"), expense_delete_choice)],
            EXPENSE_DELETE_CONFIRM_STATE: [MessageHandler(filters.Regex("^(✅ Да, удалить|❌ Отмена)$"), expense_delete_confirm)],
        },
        fallbacks=[CommandHandler("start", start)],
        allow_reentry=True
    )

    # Обработчик для напоминаний
    reminder_conv_handler = ConversationHandler(
        entry_points=[
            MessageHandler(filters.Regex("^⏰ Напоминания$"), reminder_menu),
            CommandHandler("reminders", reminder_menu)
        ],
        states={
            REMINDER_MENU_STATE: [
                MessageHandler(filters.Regex("^(📝 Добавить напоминание|📋 Список напоминаний|🗑️ Удалить напоминание|✏️ Редактировать напоминание|🔙 Назад)$"), reminder_menu),
                MessageHandler(filters.Regex("^⏰ Напоминания$"), reminder_menu)
            ],
            REMINDER_TITLE_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, reminder_title_input)],
            REMINDER_DESC_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, reminder_desc_input)],
            REMINDER_AMOUNT_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, reminder_amount_input)],
            REMINDER_START_DATE_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, reminder_start_date_input)],
            REMINDER_END_DATE_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, reminder_end_date_input)],
            REMINDER_MANAGE_STATE: [MessageHandler(filters.Regex("^(❌ Удалить \d+|🔙 Назад)$"), reminder_manage)],
            REMINDER_DELETE_STATE: [MessageHandler(filters.Regex("^(❌ Удалить \d+|🔙 Назад)$"), reminder_delete_confirm)],
            REMINDER_EDIT_CHOICE_STATE: [MessageHandler(filters.Regex("^(✏️ \d+\.|\d+\.|🔙 Назад)$"), reminder_edit_choice)],
            REMINDER_EDIT_TITLE_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, reminder_edit_title)],
            REMINDER_EDIT_DESC_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, reminder_edit_desc)],
            REMINDER_EDIT_AMOUNT_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, reminder_edit_amount)],
            REMINDER_EDIT_DATES_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, reminder_edit_dates)],
        },
        fallbacks=[CommandHandler("start", start)],
        allow_reentry=True
    )
    
    # Обработчик для планирования бюджета
    planning_conv_handler = ConversationHandler(
        entry_points=[
            MessageHandler(filters.Regex("^📅 Планирование$"), planning_menu),
            CommandHandler("planning", planning_menu)
        ],
        states={
            PLAN_MENU_STATE: [
                MessageHandler(filters.Regex("^(➕ Добавить планирование|📋 Список планов|🗑️ Удалить план|✏️ Редактировать план|🔙 Назад)$"), planning_menu),
                MessageHandler(filters.Regex("^📅 Планирование$"), planning_menu),
                MessageHandler(filters.Regex("^.* — .*$"), planning_menu)  # Обработка выбора месяца
            ],
            PLAN_MONTH_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, planning_month)],
            PLAN_TOTAL_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, planning_total)],
            PLAN_CATEGORY_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, planning_category)],
            PLAN_AMOUNT_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, planning_amount)],
            PLAN_COMMENT_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, planning_comment)],
            PLAN_DELETE_STATE: [MessageHandler(filters.Regex("^(❌ Удалить план \d+|🔙 Назад)$"), planning_delete_confirm)],
            PLAN_EDIT_CHOICE_STATE: [MessageHandler(filters.Regex("^(✏️ \d+\.|\d+\.|🔙 Назад)$"), planning_edit_choice)],
            PLAN_EDIT_MONTH_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, planning_edit_month)],
            PLAN_EDIT_TOTAL_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, planning_edit_total)],
            PLAN_EDIT_CATEGORY_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, planning_edit_category)],
            PLAN_EDIT_AMOUNT_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, planning_edit_amount)],
            PLAN_EDIT_COMMENT_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, custom_category_input)],
            PLAN_EDIT_DETAILS_STATE: [MessageHandler(filters.Regex("^(✏️ Изменить месяц|✏️ Изменить сумму|✏️ Редактировать категории|✏️ Добавить категорию|✅ Сохранить изменения|❌ Отменить)$"), planning_edit_details)],
            PLAN_EDIT_CATEGORY_CHOICE_STATE: [MessageHandler(filters.Regex("^(✏️ \d+\.|🔙 Назад)$"), planning_edit_category_choice)],
            CUSTOM_CATEGORY_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, custom_category_input)],
        },
        fallbacks=[CommandHandler("start", start)],
        allow_reentry=True
    )
    
    # Обработчик для аналитики
    analytics_conv_handler = ConversationHandler(
        entry_points=[MessageHandler(filters.Regex("^📈 Аналитика$"), analytics_menu)],
        states={
            ANALYTICS_MENU_STATE: [
                MessageHandler(filters.Regex("^(📊 Сравнение с планом|🔙 Назад)$"), analytics_month_choice)
            ],
            ANALYTICS_MONTH_STATE: [
                MessageHandler(filters.Regex("^.*\d{4}$"), analytics_month_selected),
                MessageHandler(filters.Regex("^🔙 Назад$"), analytics_month_selected)
            ]
        },
        fallbacks=[CommandHandler("start", start)],
        allow_reentry=True
    )

    # Обработчик для админ-меню
    admin_conv_handler = ConversationHandler(
        entry_points=[
            MessageHandler(filters.Regex("^(👥 Добавить пользователя|📋 Список пользователей|📁 Управление папками|🔧 Роли пользователей|📊 Статистика системы|🔙 Главное меню)$"), 
            admin_menu_handler)
        ],
        states={
            'waiting_for_username': [
                MessageHandler(filters.TEXT & ~filters.COMMAND, admin_username_input),
                MessageHandler(filters.Regex("^🔙 Назад$"), admin_back_to_menu)
            ],
            'waiting_folder_name': [
                MessageHandler(filters.TEXT & ~filters.COMMAND, admin_folder_name_input),
                MessageHandler(filters.Regex("^🔙 Назад$"), admin_back_to_menu)
            ],
            'waiting_role': [
                MessageHandler(filters.TEXT & ~filters.COMMAND, admin_role_input),
                MessageHandler(filters.Regex("^🔙 Назад$"), admin_back_to_menu)
            ]
        },
        fallbacks=[CommandHandler("start", start)],
        allow_reentry=True
    )

    application.add_handler(report_conv_handler)
    application.add_handler(correction_conv_handler)
    application.add_handler(reminder_conv_handler)
    application.add_handler(planning_conv_handler)
    application.add_handler(analytics_conv_handler)
    application.add_handler(admin_conv_handler)
    application.add_handler(CommandHandler("start", start))
    
    # Обработчик для аутентификации (должен быть перед общим обработчиком сообщений)
    application.add_handler(MessageHandler(
        filters.Regex("^🔙 Отмена$"), 
        auth_handler
    ))
    
    # Обработчик для управления группой (должен быть перед общим обработчиком сообщений)
    application.add_handler(MessageHandler(
        filters.Regex("^(👥 Участники группы|🔑 Код приглашения|📊 Статистика группы|🔙 Главное меню)$"), 
        group_management_handler
    ))
    
    # Общий обработчик сообщений (должен быть последним)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    logger.info("Бот запущен!")
    application.run_polling()

    # Функция для ежедневного обучения модели
    def daily_training():
        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute('SELECT description, category FROM expenses')
                data = cursor.fetchall()
                conn.close()

                if data:
                    logger.info(f"Данные для обучения модели: {data}")
                    descriptions = [row[0].lower() for row in data]  # Приведение описаний к нижнему регистру
                    categories = [row[1] for row in data]
                    X = vectorizer.fit_transform(descriptions)
                    classifier.fit(X, categories)
                    logger.info("Модель успешно обучена с новыми данными из базы данных.")
                else:
                    logger.warning("Нет данных для обучения модели.")
            except Exception as e:
                logger.error(f"Ошибка при обучении модели: {e}")
        else:
            logger.error("Не удалось подключиться к базе данных для обучения модели.")

    # Функция для ежедневной проверки напоминаний
    def daily_reminder_check():
        try:
            # Создаем event loop для асинхронной функции
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(check_and_send_reminders(application))
            loop.close()
            logger.info("Ежедневная проверка напоминаний завершена.")
        except Exception as e:
            logger.error(f"Ошибка при ежедневной проверке напоминаний: {e}")

    # Планирование ежедневных задач
    schedule.every().day.at("00:00").do(daily_training)
    schedule.every().day.at("09:00").do(daily_reminder_check)  # Проверяем напоминания утром

    # Запуск планировщика в отдельном потоке
    while True:
        schedule.run_pending()
        time.sleep(1)

# --- Функции планирования бюджета ---
def upsert_budget_plan(plan_month: date, total_amount: float) -> int | None:
	conn = get_db_connection()
	if not conn:
		return None
	try:
		cursor = conn.cursor()
		cursor.execute('''
			INSERT INTO budget_plans (plan_month, total_amount)
			VALUES (%s, %s)
			ON CONFLICT (plan_month)
			DO UPDATE SET total_amount = EXCLUDED.total_amount
			RETURNING id
		''', (plan_month, total_amount))
		plan_id = cursor.fetchone()[0]
		conn.commit()
		return plan_id
	except Exception as e:
		logger.error(f"Ошибка при сохранении бюджета: {e}")
		return None
	finally:
		conn.close()


def add_budget_item(plan_id: int, category: str, amount: float, comment: str | None) -> bool:
	conn = get_db_connection()
	if not conn:
		return False
	try:
		cursor = conn.cursor()
		cursor.execute('''
			INSERT INTO budget_plan_items (plan_id, category, amount, comment)
			VALUES (%s, %s, %s, %s)
		''', (plan_id, category, amount, comment))
		conn.commit()
		return True
	except Exception as e:
		logger.error(f"Ошибка при добавлении статьи бюджета: {e}")
		return False
	finally:
		conn.close()


def get_budget_plan(plan_month: date):
	conn = get_db_connection()
	if not conn:
		return None, []
	try:
		cursor = conn.cursor()
		cursor.execute('SELECT id, total_amount FROM budget_plans WHERE plan_month = %s', (plan_month,))
		row = cursor.fetchone()
		plan = None
		if row:
			plan = { 'id': row[0], 'total_amount': float(row[1]) }
			cursor.execute('SELECT category, amount, comment FROM budget_plan_items WHERE plan_id = %s ORDER BY id', (row[0],))
			items = cursor.fetchall()
		else:
			items = []
		return plan, items
	except Exception as e:
		logger.error(f"Ошибка при получении бюджета: {e}")
		return None, []
	finally:
		conn.close()

def get_budget_plan_by_month(month: int, year: int):
	"""Получить план бюджета по месяцу и году"""
	conn = get_db_connection()
	if not conn:
		return None
	try:
		cursor = conn.cursor()
		cursor.execute('''
			SELECT id, total_amount 
			FROM budget_plans 
			WHERE EXTRACT(MONTH FROM plan_month) = %s 
			AND EXTRACT(YEAR FROM plan_month) = %s
		''', (month, year))
		row = cursor.fetchone()
		if row:
			# Приводим Decimal к float для совместимости
			total_amount = float(row[1]) if row[1] is not None else 0.0
			return (row[0], total_amount)
		return None
	except Exception as e:
		logger.error(f"Ошибка при получении плана бюджета по месяцу: {e}")
		return None
	finally:
		conn.close()

# --- Диалог планирования бюджета ---
async def planning_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
	# Инициализируем пустой список для статей бюджета
	context.user_data['items'] = []
	await update.message.reply_text(
		"Выберите месяц планирования (введите в формате ММ.ГГГГ), например 08.2025:",
		reply_markup=ReplyKeyboardRemove()
	)
	return PLAN_MONTH_STATE

async def planning_month(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
	text = update.message.text.strip()
	try:
		plan_month = datetime.strptime(f"01.{text}", "%d.%m.%Y").date()
		context.user_data['plan_month'] = plan_month
		await update.message.reply_text("Введите общий бюджет на месяц (например 300000):")
		return PLAN_TOTAL_STATE
	except ValueError:
		await update.message.reply_text("Неверный формат. Введите месяц в виде ММ.ГГГГ")
		return PLAN_MONTH_STATE

async def planning_total(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
	try:
		total = float(update.message.text.replace(',', '.'))
		if total <= 0:
			raise ValueError
		context.user_data['plan_total'] = total
		await update.message.reply_text(
			"Теперь выберите категорию из списка и введите сумму. После каждого добавления можно продолжить.\n"
			"Когда закончите — отправьте 'Готово'.",
			reply_markup=get_categories_keyboard_with_done()
		)
		return PLAN_CATEGORY_STATE
	except ValueError:
		await update.message.reply_text("Введите положительное число для общего бюджета:")
		return PLAN_TOTAL_STATE

async def planning_category(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
	category = update.message.text.strip()
	if category == 'Готово':
		return await planning_summary(update, context)
	
	# Проверяем специальную кнопку для создания новой категории
	if category == "➕ Добавить новую категорию":
		await update.message.reply_text(
			"Введите название новой категории для планирования бюджета:",
			reply_markup=ReplyKeyboardRemove()
		)
		context.user_data['creating_custom_category'] = True
		return CUSTOM_CATEGORY_STATE
	
	if category not in CATEGORIES:
		await update.message.reply_text("Выберите категорию с клавиатуры, нажмите '➕ Добавить новую категорию' или отправьте 'Готово' для завершения.")
		return PLAN_CATEGORY_STATE
	
	context.user_data['current_category'] = category
	await update.message.reply_text(f"Сколько заложить на категорию '{category}'?")
	return PLAN_AMOUNT_STATE

async def planning_amount(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
	try:
		amount = float(update.message.text.replace(',', '.'))
		if amount < 0:
			raise ValueError
		context.user_data['current_amount'] = amount
		await update.message.reply_text("Добавьте комментарий к этой категории (или '-' если без комментария):")
		return PLAN_COMMENT_STATE
	except ValueError:
		await update.message.reply_text("Введите корректную сумму:")
		return PLAN_AMOUNT_STATE

async def planning_comment(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
	comment = update.message.text.strip()
	if comment == '-':
		comment = None
	items = context.user_data.get('items', [])
	items.append({
		'category': context.user_data['current_category'],
		'amount': context.user_data['current_amount'],
		'comment': comment
	})
	context.user_data['items'] = items
	await update.message.reply_text(
		"Добавлено. Можете выбрать следующую категорию или отправить 'Готово'.",
		reply_markup=get_categories_keyboard_with_done()
	)
	return PLAN_CATEGORY_STATE

async def planning_summary(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
	plan_month = context.user_data['plan_month']
	plan_total = context.user_data['plan_total']
	items = context.user_data.get('items', [])
	allocated = sum(i['amount'] for i in items)
	leftover = plan_total - allocated
	
	# Сохраняем в БД
	plan_id = upsert_budget_plan(plan_month, plan_total)
	if plan_id:
		for i in items:
			add_budget_item(plan_id, i['category'], i['amount'], i['comment'])
	
	summary_lines = [f"📅 Месяц: {plan_month.strftime('%m.%Y')}", f"💰 Общий бюджет: {plan_total:.2f}", "", "📦 Распределение:"]
	for i in items:
		comment = f" — {i['comment']}" if i['comment'] else ""
		summary_lines.append(f"- {i['category']}: {i['amount']:.2f}{comment}")
	summary_lines.append("")
	summary_lines.append(f"🧮 Сумма по категориям: {allocated:.2f}")
	summary_lines.append(f"✅ Остаток бюджета: {leftover:.2f}")
	
	await update.message.reply_text("\n".join(summary_lines), reply_markup=get_main_menu_keyboard())
	context.user_data.clear()
	return ConversationHandler.END

# --- Меню планирования ---
async def planning_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    
    # Проверяем защиту блока планирования
    if not validate_block_access("planning", user_id):
        await update.message.reply_text(
            "❌ Доступ к планированию ограничен.",
            reply_markup=get_main_menu_keyboard()
        )
        return ConversationHandler.END
    
    text = update.message.text
    if text == "📅 Планирование":
        await update.message.reply_text(
            "Выберите действие:",
            reply_markup=ReplyKeyboardMarkup([["➕ Добавить планирование", "📋 Список планов"], ["🗑️ Удалить план", "🔙 Назад"]], resize_keyboard=True)
        )
        return PLAN_MENU_STATE
    elif text == "➕ Добавить планирование":
        return await planning_start(update, context)
    elif text == "📋 Список планов":
        # Покажем краткий список месяцев с суммами
        today = datetime.now().date().replace(day=1)
        conn = get_db_connection()
        if not conn:
            await update.message.reply_text("Не удалось подключиться к БД.", reply_markup=get_main_menu_keyboard())
            return ConversationHandler.END
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT plan_month, total_amount, id FROM budget_plans ORDER BY plan_month DESC LIMIT 12')
            rows = cursor.fetchall()
        finally:
            conn.close()
        if not rows:
            await update.message.reply_text("Планы пока отсутствуют.", reply_markup=get_main_menu_keyboard())
            return ConversationHandler.END
        text_lines = ["📋 Последние планы:"]
        kb = []
        for i, (pm, total, pid) in enumerate(rows, 1):
            label = f"{pm.strftime('%m.%Y')} — {float(total):.0f}"
            text_lines.append(f"{i}. {label}")
            kb.append([KeyboardButton(label)])
        
        # Добавляем кнопки управления
        kb.append([KeyboardButton("✏️ Редактировать план"), KeyboardButton("🗑️ Удалить план")])
        kb.append([KeyboardButton("🔙 Назад")])
        
        await update.message.reply_text("\n".join(text_lines), reply_markup=ReplyKeyboardMarkup(kb, resize_keyboard=True))
        # Сохраняем планы в контексте для последующего выбора
        context.user_data['plans_list'] = rows
        return PLAN_MENU_STATE
    
    elif text == "✏️ Редактировать план":
        plans = context.user_data.get('plans_list', [])
        if not plans:
            await update.message.reply_text(
                "Список планов не найден. Попробуйте сначала открыть список планов.",
                reply_markup=get_main_menu_keyboard()
            )
            return ConversationHandler.END
        
        # Показываем список для выбора
        plans_text = "✏️ Выберите план для редактирования:\n\n"
        keyboard = []
        
        for i, (pm, total, pid) in enumerate(plans, 1):
            plans_text += f"{i}. {pm.strftime('%m.%Y')} — {float(total):.0f} Тг\n"
            keyboard.append([KeyboardButton(f"✏️ {i}. {pm.strftime('%m.%Y')}")])
        
        keyboard.append([KeyboardButton("🔙 Назад")])
        
        await update.message.reply_text(
            plans_text,
            reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
        )
        context.user_data['current_state'] = 'plan_edit_choice'
        return PLAN_EDIT_CHOICE_STATE
    
    elif text == "🗑️ Удалить план":
        return await planning_delete_start(update, context)
    elif text == "🔙 Назад":
        await update.message.reply_text("Возвращаюсь в главное меню", reply_markup=get_main_menu_keyboard())
        return ConversationHandler.END
    else:
        # Проверяем, не выбрал ли пользователь месяц из списка
        if " — " in text:
            # Это выбор месяца из списка, показываем детальный план
            return await show_detailed_plan(update, context, text)
        # Нажатие на конкретный месяц из списка — просто повторно вызвать меню
        return PLAN_MENU_STATE

async def reminder_delete_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Подтверждение удаления напоминания"""
    text = update.message.text
    
    if text == "🔙 Назад":
        await update.message.reply_text(
            "Выберите действие для напоминаний:",
            reply_markup=ReplyKeyboardMarkup([
                ["📝 Добавить напоминание", "📋 Список напоминаний"], 
                ["🗑️ Удалить напоминание", "🔙 Назад"]
            ], resize_keyboard=True)
        )
        return REMINDER_MENU_STATE
    
    # Проверяем формат "❌ Удалить N"
    if text.startswith("❌ Удалить "):
        try:
            index = int(text.split()[-1]) - 1
            reminders = context.user_data.get('reminders_list', [])
            
            if 0 <= index < len(reminders):
                reminder = reminders[index]
                rem_id = reminder[0]
                title = reminder[1]
                
                # Удаляем напоминание из БД
                if delete_reminder(rem_id):
                    await update.message.reply_text(
                        f"✅ Напоминание '{title}' успешно удалено!",
                        reply_markup=get_main_menu_keyboard()
                    )
                    return ConversationHandler.END
                else:
                    await update.message.reply_text(
                        "❌ Ошибка при удалении напоминания. Попробуйте снова.",
                        reply_markup=get_main_menu_keyboard()
                    )
                    return ConversationHandler.END
            else:
                await update.message.reply_text(
                    "❌ Неверный номер напоминания. Попробуйте снова.",
                    reply_markup=get_main_menu_keyboard()
                )
                return ConversationHandler.END
        except (ValueError, IndexError):
            await update.message.reply_text(
                "❌ Неверный формат. Попробуйте снова.",
                reply_markup=get_main_menu_keyboard()
            )
            return ConversationHandler.END
    
    await update.message.reply_text(
        "❌ Неверный выбор. Попробуйте снова.",
        reply_markup=get_main_menu_keyboard()
    )
    return ConversationHandler.END

async def planning_edit_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора плана для редактирования"""
    text = update.message.text
    
    if text == "🔙 Назад":
        context.user_data.pop('current_state', None)
        await update.message.reply_text(
            "Выберите действие:",
            reply_markup=ReplyKeyboardMarkup([["➕ Добавить планирование", "📋 Список планов"], ["🗑️ Удалить план", "🔙 Назад"]], resize_keyboard=True)
        )
        return PLAN_MENU_STATE
    
    # Парсим выбор пользователя
    if text.startswith("✏️ "):
        try:
            # Извлекаем номер из текста "✏️ 1. ММ.ГГГГ"
            choice_num = int(text.split(".")[0].split()[-1]) - 1
            plans = context.user_data.get('plans_list', [])
            
            if 0 <= choice_num < len(plans):
                selected_plan = plans[choice_num]
                context.user_data['editing_plan'] = selected_plan
                
                pm, total, pid = selected_plan
                
                # Загружаем детали плана
                conn = get_db_connection()
                if not conn:
                    await update.message.reply_text(
                        "❌ Ошибка подключения к базе данных.",
                        reply_markup=get_main_menu_keyboard()
                    )
                    return ConversationHandler.END
                
                try:
                    cursor = conn.cursor()
                    cursor.execute('SELECT category, amount, comment FROM budget_plan_items WHERE plan_id = %s ORDER BY id', (pid,))
                    current_items = cursor.fetchall()
                    
                    context.user_data['current_plan_items'] = current_items
                    context.user_data['editing_items'] = list(current_items)  # Копия для редактирования
                    
                    # Показываем детали плана с кнопками управления
                    plan_details = f"✏️ Редактирование плана:\n\n"
                    plan_details += f"📅 Месяц: {pm.strftime('%m.%Y')}\n"
                    plan_details += f"💰 Общая сумма: {float(total):.2f} Тг\n\n"
                    plan_details += f"📋 Категории:\n"
                    
                    for i, (cat, amt, comm) in enumerate(current_items, 1):
                        plan_details += f"{i}. {cat}: {float(amt):.2f} Тг\n"
                    
                    keyboard = [
                        ["✏️ Изменить месяц", "✏️ Изменить сумму"],
                        ["✏️ Редактировать категории", "✏️ Добавить категорию"],
                        ["✅ Сохранить изменения", "❌ Отменить"]
                    ]
                    
                    await update.message.reply_text(
                        plan_details,
                        reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
                    )
                    return PLAN_EDIT_DETAILS_STATE
                    
                except Exception as e:
                    logger.error(f"Ошибка при загрузке плана: {e}")
                    await update.message.reply_text(
                        "❌ Ошибка при загрузке плана.",
                        reply_markup=get_main_menu_keyboard()
                    )
                    return ConversationHandler.END
                finally:
                    conn.close()
            else:
                await update.message.reply_text(
                    "❌ Неверный выбор. Попробуйте снова.",
                    reply_markup=ReplyKeyboardMarkup([["🔙 Назад"]], resize_keyboard=True)
                )
                return PLAN_EDIT_CHOICE_STATE
        except (ValueError, IndexError):
            await update.message.reply_text(
                "❌ Неверный формат. Попробуйте снова.",
                reply_markup=ReplyKeyboardMarkup([["🔙 Назад"]], resize_keyboard=True)
            )
            return PLAN_EDIT_CHOICE_STATE
    
    # Обрабатываем выбор по номеру (например "1. 09.2025")
    elif text and text[0].isdigit() and "." in text:
        try:
            # Извлекаем номер из текста "1. ММ.ГГГГ"
            choice_num = int(text.split(".")[0]) - 1
            plans = context.user_data.get('plans_list', [])
            
            if 0 <= choice_num < len(plans):
                selected_plan = plans[choice_num]
                context.user_data['editing_plan'] = selected_plan
                
                pm, total, pid = selected_plan
                
                # Загружаем детали плана
                conn = get_db_connection()
                if not conn:
                    await update.message.reply_text(
                        "❌ Ошибка подключения к базе данных.",
                        reply_markup=get_main_menu_keyboard()
                    )
                    return ConversationHandler.END
                
                try:
                    cursor = conn.cursor()
                    cursor.execute('SELECT category, amount, comment FROM budget_plan_items WHERE plan_id = %s ORDER BY id', (pid,))
                    current_items = cursor.fetchall()
                    
                    context.user_data['current_plan_items'] = current_items
                    context.user_data['editing_items'] = list(current_items)  # Копия для редактирования
                    
                    # Показываем детали плана с кнопками управления
                    plan_details = f"✏️ Редактирование плана:\n\n"
                    plan_details += f"📅 Месяц: {pm.strftime('%m.%Y')}\n"
                    plan_details += f"💰 Общая сумма: {float(total):.2f} Тг\n\n"
                    plan_details += f"📋 Категории:\n"
                    
                    for i, (cat, amt, comm) in enumerate(current_items, 1):
                        plan_details += f"{i}. {cat}: {float(amt):.2f} Тг\n"
                    
                    keyboard = [
                        ["✏️ Изменить месяц", "✏️ Изменить сумму"],
                        ["✏️ Редактировать категории", "✏️ Добавить категорию"],
                        ["✅ Сохранить изменения", "❌ Отменить"]
                    ]
                    
                    await update.message.reply_text(
                        plan_details,
                        reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
                    )
                    return PLAN_EDIT_DETAILS_STATE
                    
                except Exception as e:
                    logger.error(f"Ошибка при загрузке плана: {e}")
                    await update.message.reply_text(
                        "❌ Ошибка при загрузке плана.",
                        reply_markup=get_main_menu_keyboard()
                    )
                    return ConversationHandler.END
                finally:
                    conn.close()
            else:
                await update.message.reply_text(
                    "❌ Неверный выбор. Попробуйте снова.",
                    reply_markup=ReplyKeyboardMarkup([["🔙 Назад"]], resize_keyboard=True)
                )
                return PLAN_EDIT_CHOICE_STATE
        except (ValueError, IndexError):
            await update.message.reply_text(
                "❌ Неверный формат. Попробуйте снова.",
                reply_markup=ReplyKeyboardMarkup([["🔙 Назад"]], resize_keyboard=True)
            )
            return PLAN_EDIT_CHOICE_STATE
    
    return PLAN_EDIT_CHOICE_STATE

async def planning_edit_month(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Редактирование месяца плана"""
    text = update.message.text.strip()
    
    try:
        if text and text != context.user_data.get('editing_plan', [None])[0].strftime('%m.%Y'):
            # Пользователь ввел новый месяц
            new_month = datetime.strptime(f"01.{text}", "%d.%m.%Y").date()
        else:
            # Пользователь оставил текущий месяц
            selected_plan = context.user_data.get('editing_plan')
            pm, total, pid = selected_plan
            new_month = pm
    except ValueError:
        await update.message.reply_text(
            "❌ Неверный формат. Введите месяц в виде ММ.ГГГГ:",
            reply_markup=ReplyKeyboardRemove()
        )
        return PLAN_EDIT_MONTH_STATE
    
    context.user_data['new_plan_month'] = new_month
    
    # Обновляем план в контексте
    selected_plan = context.user_data.get('editing_plan')
    pm, total, pid = selected_plan
    context.user_data['editing_plan'] = (new_month, total, pid)
    
    # Возвращаемся к детальному редактированию
    editing_items = context.user_data.get('editing_items', [])
    
    plan_details = f"✏️ Редактирование плана:\n\n"
    plan_details += f"📅 Месяц: {new_month.strftime('%m.%Y')}\n"
    plan_details += f"💰 Общая сумма: {float(total):.2f} Тг\n\n"
    plan_details += f"📋 Категории:\n"
    
    for i, (cat, amt, comm) in enumerate(editing_items, 1):
        plan_details += f"{i}. {cat}: {float(amt):.2f} Тг\n"
    
    keyboard = [
        ["✏️ Изменить месяц", "✏️ Изменить сумму"],
        ["✏️ Редактировать категории", "✏️ Добавить категорию"],
        ["✅ Сохранить изменения", "❌ Отменить"]
    ]
    
    await update.message.reply_text(
        f"✅ Месяц обновлен: {new_month.strftime('%m.%Y')}\n\n" + plan_details,
        reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    )
    return PLAN_EDIT_DETAILS_STATE

async def planning_edit_total(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Редактирование общей суммы плана"""
    text = update.message.text.strip()
    
    try:
        if text:
            new_total = float(text.replace(',', '.'))
            if new_total <= 0:
                raise ValueError
        else:
            # Пользователь оставил текущую сумму
            selected_plan = context.user_data.get('editing_plan')
            pm, total, pid = selected_plan
            new_total = float(total)
    except ValueError:
        await update.message.reply_text(
            "❌ Введите положительное число для общего бюджета:",
            reply_markup=ReplyKeyboardRemove()
        )
        return PLAN_EDIT_TOTAL_STATE
    
    context.user_data['new_plan_total'] = new_total
    
    # Обновляем план в контексте
    selected_plan = context.user_data.get('editing_plan')
    pm, total, pid = selected_plan
    context.user_data['editing_plan'] = (pm, new_total, pid)
    
    # Возвращаемся к детальному редактированию
    editing_items = context.user_data.get('editing_items', [])
    
    plan_details = f"✏️ Редактирование плана:\n\n"
    plan_details += f"📅 Месяц: {pm.strftime('%m.%Y')}\n"
    plan_details += f"💰 Общая сумма: {new_total:.2f} Тг\n\n"
    plan_details += f"📋 Категории:\n"
    
    for i, (cat, amt, comm) in enumerate(editing_items, 1):
        plan_details += f"{i}. {cat}: {float(amt):.2f} Тг\n"
    
    keyboard = [
        ["✏️ Изменить месяц", "✏️ Изменить сумму"],
        ["✏️ Редактировать категории", "✏️ Добавить категорию"],
        ["✅ Сохранить изменения", "❌ Отменить"]
    ]
    
    await update.message.reply_text(
        f"✅ Сумма обновлена: {new_total:.2f} Тг\n\n" + plan_details,
        reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    )
    return PLAN_EDIT_DETAILS_STATE

async def planning_edit_category(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Редактирование категории плана"""
    category = update.message.text.strip()
    
    if category == 'Готово':
        return await planning_edit_save(update, context)
    
    # Проверяем специальную кнопку для создания новой категории
    if category == "➕ Добавить новую категорию":
        await update.message.reply_text(
            "Введите название новой категории для планирования бюджета:",
            reply_markup=ReplyKeyboardRemove()
        )
        return PLAN_EDIT_COMMENT_STATE
    
    if category not in CATEGORIES:
        await update.message.reply_text(
            "Выберите категорию с клавиатуры, нажмите '➕ Добавить новую категорию' или отправьте 'Готово' для завершения.",
            reply_markup=get_categories_keyboard_with_done()
        )
        return PLAN_EDIT_CATEGORY_STATE
    
    context.user_data['editing_category'] = category
    
    # Проверяем, есть ли уже эта категория в плане
    editing_items = context.user_data.get('editing_items', [])
    existing_amount = 0
    for i, (cat, amt, comm) in enumerate(editing_items):
        if cat == category:
            existing_amount = float(amt)
            break
    
    if existing_amount > 0:
        await update.message.reply_text(
            f"Категория '{category}' уже есть в плане с суммой {existing_amount:.2f} Тг.\n"
            f"Введите новую сумму для этой категории:",
            reply_markup=ReplyKeyboardRemove()
        )
    else:
        await update.message.reply_text(
            f"Введите сумму для категории '{category}':",
            reply_markup=ReplyKeyboardRemove()
        )
    
    return PLAN_EDIT_AMOUNT_STATE

async def planning_edit_amount(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Редактирование суммы категории плана"""
    try:
        amount = float(update.message.text.replace(',', '.'))
        if amount < 0:
            raise ValueError
    except ValueError:
        await update.message.reply_text(
            "❌ Введите корректную сумму:",
            reply_markup=ReplyKeyboardRemove()
        )
        return PLAN_EDIT_AMOUNT_STATE
    
    category = context.user_data.get('editing_category')
    editing_items = context.user_data.get('editing_items', [])
    
    # Обновляем или добавляем категорию
    updated = False
    for i, (cat, amt, comm) in enumerate(editing_items):
        if cat == category:
            editing_items[i] = (cat, amount, comm)
            updated = True
            break
    
    if not updated:
        editing_items.append((category, amount, None))
    
    context.user_data['editing_items'] = editing_items
    
    # Проверяем, редактируем ли мы существующую категорию или добавляем новую
    if 'editing_category_index' in context.user_data:
        # Редактируем существующую категорию
        editing_items[context.user_data['editing_category_index']] = (category, amount, comm)
        context.user_data.pop('editing_category_index', None)
        context.user_data.pop('editing_category_item', None)
        
        # Возвращаемся к детальному редактированию
        selected_plan = context.user_data.get('editing_plan')
        pm, total, pid = selected_plan
        
        plan_details = f"✏️ Редактирование плана:\n\n"
        plan_details += f"📅 Месяц: {pm.strftime('%m.%Y')}\n"
        plan_details += f"💰 Общая сумма: {float(total):.2f} Тг\n\n"
        plan_details += f"📋 Категории:\n"
        
        for i, (cat, amt, comm) in enumerate(editing_items, 1):
            plan_details += f"{i}. {cat}: {float(amt):.2f} Тг\n"
        
        keyboard = [
            ["✏️ Изменить месяц", "✏️ Изменить сумму"],
            ["✏️ Редактировать категории", "✏️ Добавить категорию"],
            ["✅ Сохранить изменения", "❌ Отменить"]
        ]
        
        await update.message.reply_text(
            f"✅ {category}: {amount:.2f} Тг\n\n" + plan_details,
            reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
        )
        return PLAN_EDIT_DETAILS_STATE
    else:
        # Добавляем новую категорию
        await update.message.reply_text(
            f"✅ {category}: {amount:.2f} Тг\n\n"
            f"Выберите следующую категорию или 'Готово' для завершения:",
            reply_markup=get_categories_keyboard_with_done()
        )
        return PLAN_EDIT_CATEGORY_STATE

async def planning_edit_save(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Сохранение отредактированного плана"""
    selected_plan = context.user_data.get('editing_plan')
    pm, total, pid = selected_plan
    
    new_month = context.user_data.get('new_plan_month', pm)
    new_total = context.user_data.get('new_plan_total', float(total))
    editing_items = context.user_data.get('editing_items', [])
    
    # Сохраняем изменения в базу данных
    conn = get_db_connection()
    if not conn:
        await update.message.reply_text(
            "❌ Ошибка подключения к базе данных.",
            reply_markup=get_main_menu_keyboard()
        )
        return ConversationHandler.END
    
    try:
        cursor = conn.cursor()
        
        # Обновляем основной план
        cursor.execute('''
            UPDATE budget_plans 
            SET plan_month = %s, total_amount = %s 
            WHERE id = %s
        ''', (new_month, new_total, pid))
        
        # Удаляем старые элементы плана
        cursor.execute('DELETE FROM budget_plan_items WHERE plan_id = %s', (pid,))
        
        # Добавляем новые элементы плана
        for category, amount, comment in editing_items:
            cursor.execute('''
                INSERT INTO budget_plan_items (plan_id, category, amount, comment)
                VALUES (%s, %s, %s, %s)
            ''', (pid, category, amount, comment))
        
        conn.commit()
        
        # Формируем итоговое сообщение
        items_text = "\n".join([f"• {cat}: {float(amt):.2f} Тг" for cat, amt, comm in editing_items])
        
        await update.message.reply_text(
            f"✅ План успешно обновлен!\n\n"
            f"📅 Месяц: {new_month.strftime('%m.%Y')}\n"
            f"💰 Общая сумма: {new_total:.2f} Тг\n\n"
            f"📋 Категории:\n{items_text}",
            reply_markup=get_main_menu_keyboard()
        )
        
    except Exception as e:
        logger.error(f"Ошибка при сохранении плана: {e}")
        await update.message.reply_text(
            "❌ Ошибка при сохранении плана. Попробуйте снова.",
            reply_markup=get_main_menu_keyboard()
        )
    finally:
        conn.close()
    
    # Очищаем контекст
    context.user_data.pop('editing_plan', None)
    context.user_data.pop('new_plan_month', None)
    context.user_data.pop('new_plan_total', None)
    context.user_data.pop('editing_items', None)
    context.user_data.pop('editing_category', None)
    context.user_data.pop('current_plan_items', None)
    context.user_data.pop('current_state', None)
    
    return ConversationHandler.END

async def planning_edit_details(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка детального редактирования плана"""
    text = update.message.text
    
    if text == "✏️ Изменить месяц":
        selected_plan = context.user_data.get('editing_plan')
        pm, total, pid = selected_plan
        
        await update.message.reply_text(
            f"📅 Текущий месяц: {pm.strftime('%m.%Y')}\n\n"
            f"Введите новый месяц в формате ММ.ГГГГ:",
            reply_markup=ReplyKeyboardRemove()
        )
        return PLAN_EDIT_MONTH_STATE
    
    elif text == "✏️ Изменить сумму":
        selected_plan = context.user_data.get('editing_plan')
        pm, total, pid = selected_plan
        
        await update.message.reply_text(
            f"💰 Текущая сумма: {float(total):.2f} Тг\n\n"
            f"Введите новую общую сумму бюджета:",
            reply_markup=ReplyKeyboardRemove()
        )
        return PLAN_EDIT_TOTAL_STATE
    
    elif text == "✏️ Редактировать категории":
        editing_items = context.user_data.get('editing_items', [])
        
        if not editing_items:
            await update.message.reply_text(
                "❌ В плане нет категорий для редактирования.",
                reply_markup=ReplyKeyboardMarkup([["🔙 Назад"]], resize_keyboard=True)
            )
            return PLAN_EDIT_DETAILS_STATE
        
        # Показываем список категорий для редактирования
        categories_text = "✏️ Выберите категорию для редактирования:\n\n"
        keyboard = []
        
        for i, (cat, amt, comm) in enumerate(editing_items, 1):
            categories_text += f"{i}. {cat}: {float(amt):.2f} Тг\n"
            keyboard.append([KeyboardButton(f"✏️ {i}. {cat}")])
        
        keyboard.append([KeyboardButton("🔙 Назад")])
        
        await update.message.reply_text(
            categories_text,
            reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
        )
        return PLAN_EDIT_CATEGORY_CHOICE_STATE
    
    elif text == "✏️ Добавить категорию":
        await update.message.reply_text(
            "Выберите категорию для добавления:",
            reply_markup=get_categories_keyboard_with_done()
        )
        return PLAN_EDIT_CATEGORY_STATE
    
    elif text == "✅ Сохранить изменения":
        return await planning_edit_save(update, context)
    
    elif text == "❌ Отменить":
        context.user_data.pop('editing_plan', None)
        context.user_data.pop('current_plan_items', None)
        context.user_data.pop('editing_items', None)
        context.user_data.pop('current_state', None)
        
        await update.message.reply_text(
            "❌ Редактирование отменено.",
            reply_markup=get_main_menu_keyboard()
        )
        return ConversationHandler.END
    
    return PLAN_EDIT_DETAILS_STATE

async def planning_edit_category_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Выбор категории для редактирования"""
    text = update.message.text
    
    if text == "🔙 Назад":
        # Возвращаемся к детальному редактированию
        selected_plan = context.user_data.get('editing_plan')
        pm, total, pid = selected_plan
        editing_items = context.user_data.get('editing_items', [])
        
        plan_details = f"✏️ Редактирование плана:\n\n"
        plan_details += f"📅 Месяц: {pm.strftime('%m.%Y')}\n"
        plan_details += f"💰 Общая сумма: {float(total):.2f} Тг\n\n"
        plan_details += f"📋 Категории:\n"
        
        for i, (cat, amt, comm) in enumerate(editing_items, 1):
            plan_details += f"{i}. {cat}: {float(amt):.2f} Тг\n"
        
        keyboard = [
            ["✏️ Изменить месяц", "✏️ Изменить сумму"],
            ["✏️ Редактировать категории", "✏️ Добавить категорию"],
            ["✅ Сохранить изменения", "❌ Отменить"]
        ]
        
        await update.message.reply_text(
            plan_details,
            reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
        )
        return PLAN_EDIT_DETAILS_STATE
    
    # Парсим выбор категории
    if text.startswith("✏️ "):
        try:
            choice_num = int(text.split(".")[0].split()[-1]) - 1
            editing_items = context.user_data.get('editing_items', [])
            
            if 0 <= choice_num < len(editing_items):
                selected_category = editing_items[choice_num]
                context.user_data['editing_category_item'] = selected_category
                context.user_data['editing_category_index'] = choice_num
                
                cat, amt, comm = selected_category
                
                await update.message.reply_text(
                    f"✏️ Редактирование категории:\n\n"
                    f"📝 Категория: {cat}\n"
                    f"💰 Сумма: {float(amt):.2f} Тг\n"
                    f"📄 Комментарий: {comm or 'Не указан'}\n\n"
                    f"Введите новую сумму для этой категории:",
                    reply_markup=ReplyKeyboardRemove()
                )
                return PLAN_EDIT_AMOUNT_STATE
            else:
                await update.message.reply_text(
                    "❌ Неверный выбор. Попробуйте снова.",
                    reply_markup=ReplyKeyboardMarkup([["🔙 Назад"]], resize_keyboard=True)
                )
                return PLAN_EDIT_CATEGORY_CHOICE_STATE
        except (ValueError, IndexError):
            await update.message.reply_text(
                "❌ Неверный формат. Попробуйте снова.",
                reply_markup=ReplyKeyboardMarkup([["🔙 Назад"]], resize_keyboard=True)
            )
            return PLAN_EDIT_CATEGORY_CHOICE_STATE
    
    return PLAN_EDIT_CATEGORY_CHOICE_STATE

async def planning_delete_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начало удаления плана бюджета"""
    text = update.message.text
    
    if text == "🗑️ Удалить план":
        # Получаем список планов
        conn = get_db_connection()
        if not conn:
            await update.message.reply_text("Не удалось подключиться к БД.", reply_markup=get_main_menu_keyboard())
            return ConversationHandler.END
        
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT id, plan_month, total_amount FROM budget_plans ORDER BY plan_month DESC')
            plans = cursor.fetchall()
        finally:
            conn.close()
        
        if not plans:
            await update.message.reply_text(
                "У вас нет планов бюджета для удаления.",
                reply_markup=get_main_menu_keyboard()
            )
            return ConversationHandler.END
        
        plans_text = "🗑️ Выберите план для удаления:\n\n"
        keyboard = []
        
        for i, (plan_id, plan_month, total_amount) in enumerate(plans, 1):
            plans_text += f"{i}. {plan_month.strftime('%m.%Y')} - {float(total_amount):.0f} Тг\n"
            keyboard.append([KeyboardButton(f"❌ Удалить план {i}")])
        
        keyboard.append([KeyboardButton("🔙 Назад")])
        
        await update.message.reply_text(
            plans_text,
            reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
        )
        context.user_data['plans_list'] = plans
        return PLAN_DELETE_STATE
    
    return PLAN_MENU_STATE

async def planning_delete_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Подтверждение удаления плана бюджета"""
    text = update.message.text
    
    if text == "🔙 Назад":
        await update.message.reply_text(
            "Выберите действие:",
            reply_markup=ReplyKeyboardMarkup([["➕ Добавить планирование", "📋 Список планов"], ["🔙 Назад"]], resize_keyboard=True)
        )
        return PLAN_MENU_STATE
    
    # Проверяем формат "❌ Удалить план N"
    if text.startswith("❌ Удалить план "):
        try:
            index = int(text.split()[-1]) - 1
            plans = context.user_data.get('plans_list', [])
            
            if 0 <= index < len(plans):
                plan = plans[index]
                plan_id = plan[0]
                plan_month = plan[1]
                total_amount = plan[2]
                
                # Удаляем план из БД
                if delete_budget_plan(plan_id):
                    await update.message.reply_text(
                        f"✅ План на {plan_month.strftime('%m.%Y')} ({float(total_amount):.0f} Тг) успешно удален!",
                        reply_markup=get_main_menu_keyboard()
                    )
                    return ConversationHandler.END
                else:
                    await update.message.reply_text(
                        "❌ Ошибка при удалении плана. Попробуйте снова.",
                        reply_markup=get_main_menu_keyboard()
                    )
                    return ConversationHandler.END
            else:
                await update.message.reply_text(
                    "❌ Неверный номер плана. Попробуйте снова.",
                    reply_markup=get_main_menu_keyboard()
                )
                return ConversationHandler.END
        except (ValueError, IndexError):
            await update.message.reply_text(
                "❌ Неверный формат. Попробуйте снова.",
                reply_markup=get_main_menu_keyboard()
            )
            return ConversationHandler.END
    
    await update.message.reply_text(
        "❌ Неверный выбор. Попробуйте снова.",
        reply_markup=get_main_menu_keyboard()
    )
    return ConversationHandler.END

def delete_budget_plan(plan_id):
    """Удалить план бюджета и все его статьи"""
    conn = get_db_connection()
    if not conn:
        return False
    try:
        cursor = conn.cursor()
        # Сначала удаляем все статьи плана
        cursor.execute('DELETE FROM budget_plan_items WHERE plan_id = %s', (plan_id,))
        # Затем удаляем сам план
        cursor.execute('DELETE FROM budget_plans WHERE id = %s', (plan_id,))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Ошибка при удалении плана бюджета: {e}")
        return False
    finally:
        conn.close()

async def custom_category_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ввода новой пользовательской категории"""
    new_category = update.message.text.strip()
    
    if not new_category:
        await update.message.reply_text(
            "Название категории не может быть пустым. Попробуйте снова:",
            reply_markup=get_categories_keyboard()
        )
        return CATEGORY_CHOICE_STATE
    
    # Проверяем, не существует ли уже такая категория
    if new_category in CATEGORIES:
        await update.message.reply_text(
            f"Категория '{new_category}' уже существует. Выберите её из списка или введите другую:",
            reply_markup=get_categories_keyboard()
        )
        return CATEGORY_CHOICE_STATE
    
    # Добавляем новую категорию в словарь
    CATEGORIES[new_category] = []
    
    # Проверяем, из какого контекста мы пришли
    if context.user_data.get('creating_custom_category'):
        # Мы в процессе планирования бюджета
        context.user_data['current_category'] = new_category
        context.user_data.pop('creating_custom_category', None)  # Убираем флаг
        await update.message.reply_text(
            f"✅ Создана новая категория '{new_category}'!\n\n"
            f"Сколько заложить на категорию '{new_category}'?",
            reply_markup=ReplyKeyboardRemove()
        )
        return PLAN_AMOUNT_STATE
    else:
        # Мы в процессе исправления категории
        selected_expense = context.user_data.get('selected_expense')
        if not selected_expense:
            await update.message.reply_text(
                "Ошибка: расход не найден. Попробуйте снова.",
                reply_markup=get_main_menu_keyboard()
            )
            return ConversationHandler.END
        
        exp_id, amount, desc, old_cat, date = selected_expense
        
        # Обновляем категорию в базе данных
        if update_expense_category(exp_id, new_category):
            context.user_data['selected_expense'] = (exp_id, amount, desc, new_category, date)
            await update.message.reply_text(
                f"✅ Создана новая категория '{new_category}' и применена к расходу!\n\n"
                f"📝 {desc}\n"
                f"🏷️ Новая категория: {new_category}\n"
                f"💰 Сумма: {amount} Тг\n\n"
                f"Теперь введите новую сумму (например: 1500.50) или отправьте текущую сумму без изменений:",
                reply_markup=ReplyKeyboardRemove()
            )
            return AMOUNT_EDIT_STATE
        else:
            await update.message.reply_text(
                "❌ Ошибка при обновлении категории. Попробуйте снова.",
                reply_markup=get_main_menu_keyboard()
            )
            return ConversationHandler.END

async def show_detailed_plan(update: Update, context: ContextTypes.DEFAULT_TYPE, month_text: str) -> int:
    """Показывает детальный план бюджета с круговой диаграммой"""
    try:
        # Извлекаем месяц и год из текста (формат: "08.2025 — 50000")
        month_part = month_text.split(" — ")[0]
        month, year = month_part.split(".")
        plan_date = datetime.strptime(f"01.{month}.{year}", "%d.%m.%Y").date()
        
        # Получаем план из базы данных
        conn = get_db_connection()
        if not conn:
            await update.message.reply_text("Не удалось подключиться к БД.", reply_markup=get_main_menu_keyboard())
            return ConversationHandler.END
        
        try:
            cursor = conn.cursor()
            # Получаем основной план
            cursor.execute('SELECT id, total_amount FROM budget_plans WHERE plan_month = %s', (plan_date,))
            plan_row = cursor.fetchone()
            
            if not plan_row:
                await update.message.reply_text(f"План на {month_part} не найден.", reply_markup=get_main_menu_keyboard())
                return ConversationHandler.END
            
            plan_id, total_amount = plan_row
            
            # Получаем статьи бюджета
            cursor.execute('SELECT category, amount, comment FROM budget_plan_items WHERE plan_id = %s ORDER BY amount DESC', (plan_id,))
            items = cursor.fetchall()
            
        finally:
            conn.close()
        
        if not items:
            await update.message.reply_text(
                f"📋 План на {month_part}\n"
                f"💰 Общий бюджет: {float(total_amount):.0f} Тг\n"
                f"📝 Статьи бюджета не добавлены.",
                reply_markup=ReplyKeyboardMarkup([["🔙 Назад"]], resize_keyboard=True)
            )
            return PLAN_MENU_STATE
        
        # Создаем круговую диаграмму
        categories = [item[0] for item in items]
        amounts = [float(item[1]) for item in items]
        comments = [item[2] for item in items]
        
        # Создаем фигуру
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Создаем круговую диаграмму
        wedges, texts, autotexts = ax.pie(
            amounts, 
            labels=categories, 
            autopct='%1.1f%%',
            startangle=90,
            shadow=True,
            explode=[0.05] * len(amounts),
            colors=['#6B8E23', '#4682B4', '#CD853F', '#20B2AA', '#8A2BE2', '#32CD32', '#FF8C00', '#DC143C', '#1E90FF', '#9370DB']
        )
        
        # Настройка текста
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title(f'📊 Распределение бюджета на {month_part}', fontsize=16, fontweight='bold', pad=20)
        
        # Добавляем легенду
        legend_texts = []
        for i, (cat, amt, comm) in enumerate(zip(categories, amounts, comments)):
            legend_text = f"{cat}: {amt:.0f} Тг"
            if comm:
                legend_text += f" ({comm})"
            legend_texts.append(legend_text)
        
        ax.legend(legend_texts, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
        
        plt.tight_layout()
        
        # Сохраняем график в буфер
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        # Формируем текст с детальной информацией
        total_allocated = sum(amounts)
        remaining = float(total_amount) - total_allocated
        
        detail_text = f"📋 Детальный план на {month_part}\n\n"
        detail_text += f"💰 Общий бюджет: {float(total_amount):.0f} Тг\n"
        detail_text += f"📊 Распределено: {total_allocated:.0f} Тг\n"
        detail_text += f"💵 Остаток: {remaining:.0f} Тг\n\n"
        detail_text += "📝 Статьи бюджета:\n"
        
        for i, (cat, amt, comm) in enumerate(zip(categories, amounts, comments), 1):
            detail_text += f"{i}. {cat}: {amt:.0f} Тг"
            if comm:
                detail_text += f" ({comm})"
            detail_text += "\n"
        
        # Отправляем график с детальной информацией
        await update.message.reply_photo(
            photo=buf,
            caption=detail_text,
            reply_markup=ReplyKeyboardMarkup([["🔙 Назад"]], resize_keyboard=True)
        )
        
        return PLAN_MENU_STATE
        
    except Exception as e:
        logger.error(f"Ошибка при показе детального плана: {e}")
        await update.message.reply_text(
            f"Произошла ошибка при показе плана: {e}",
            reply_markup=get_main_menu_keyboard()
        )
        return ConversationHandler.END

async def expense_delete_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Выбор расхода для удаления"""
    try:
        choice = int(update.message.text)
        expenses = context.user_data.get('expenses_to_delete', [])
        
        if choice < 1 or choice > len(expenses):
            await update.message.reply_text(
                f"Пожалуйста, введите число от 1 до {len(expenses)}",
                reply_markup=get_main_menu_keyboard()
            )
            return ConversationHandler.END
        
        # Сохраняем выбранный расход для удаления
        selected_expense = expenses[choice - 1]
        context.user_data['expense_to_delete'] = selected_expense
        
        exp_id, amount, desc, cat, date = selected_expense
        date_str = date.strftime("%d.%m.%Y") if date else "Неизвестно"
        
        # Создаем клавиатуру для подтверждения
        keyboard = [
            [KeyboardButton("✅ Да, удалить")],
            [KeyboardButton("❌ Отмена")]
        ]
        reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)
        
        await update.message.reply_text(
            f"⚠️ Вы действительно хотите удалить этот расход?\n\n"
            f"📝 {desc}\n"
            f"💰 {amount} Тг\n"
            f"🏷️ Категория: {cat}\n"
            f"📅 {date_str}\n\n"
            f"Это действие нельзя отменить!",
            reply_markup=reply_markup
        )
        return EXPENSE_DELETE_CONFIRM_STATE
        
    except ValueError:
        await update.message.reply_text(
            "Пожалуйста, введите число.",
            reply_markup=get_main_menu_keyboard()
        )
        return ConversationHandler.END

async def expense_delete_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Подтверждение удаления расхода"""
    choice = update.message.text.strip()
    
    if choice == "✅ Да, удалить":
        selected_expense = context.user_data.get('expense_to_delete')
        if not selected_expense:
            await update.message.reply_text(
                "Ошибка: расход не найден. Попробуйте снова.",
                reply_markup=get_main_menu_keyboard()
            )
            return ConversationHandler.END
        
        exp_id, amount, desc, cat, date = selected_expense
        
        # Удаляем расход из базы данных
        if delete_expense(exp_id):
            await update.message.reply_text(
                f"✅ Расход успешно удален!\n\n"
                f"📝 {desc}\n"
                f"💰 {amount} Тг\n"
                f"🏷️ {cat}\n"
                f"📅 {date.strftime('%d.%m.%Y') if date else 'Неизвестно'}",
                reply_markup=get_main_menu_keyboard()
            )
        else:
            await update.message.reply_text(
                "❌ Ошибка при удалении расхода. Попробуйте снова.",
                reply_markup=get_main_menu_keyboard()
            )
        return ConversationHandler.END
        
    elif choice == "❌ Отмена":
        await update.message.reply_text(
            "Удаление отменено. Возвращаемся в главное меню.",
            reply_markup=get_main_menu_keyboard()
        )
        return ConversationHandler.END
    
    else:
        await update.message.reply_text(
            "Пожалуйста, выберите один из предложенных вариантов.",
            reply_markup=get_main_menu_keyboard()
        )
        return ConversationHandler.END

def get_monthly_expenses(month: int, year: int):
    """Получить расходы за месяц"""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT category, SUM(amount) as total
            FROM expenses 
            WHERE EXTRACT(MONTH FROM transaction_date) = %s
            AND EXTRACT(YEAR FROM transaction_date) = %s
            GROUP BY category
            ORDER BY total DESC
        ''', (month, year))
        
        # Приводим все суммы к float для совместимости
        rows = cursor.fetchall()
        return [(row[0], float(row[1])) for row in rows]
    except Exception as e:
        logger.error(f"Ошибка при получении расходов за месяц: {e}")
        return []
    finally:
        conn.close()

def get_budget_plan_items(plan_id: int):
    """Получить элементы плана бюджета"""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT category, amount, comment
            FROM budget_plan_items 
            WHERE plan_id = %s
            ORDER BY amount DESC
        ''', (plan_id,))
        
        # Приводим все суммы к float для совместимости
        rows = cursor.fetchall()
        return [(row[0], float(row[1]), row[2]) for row in rows]
    except Exception as e:
        logger.error(f"Ошибка при получении элементов плана: {e}")
        return []
    finally:
        conn.close()

# --- АДМИН-ФУНКЦИОНАЛЬНОСТЬ ---
# Константы для админ-системы
ADMIN_USER_ID = 498410375  # Замените на ваш Telegram ID
USERS_FILE = "authorized_users.json"

# Роли пользователей
USER_ROLES = {
    "admin": "Администратор",
    "moderator": "Модератор", 
    "user": "Пользователь"
}

def load_authorized_users():
    """Загружает список авторизованных пользователей из файла"""
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"users": [], "admin": ADMIN_USER_ID}
    except Exception as e:
        logger.error(f"Ошибка при загрузке пользователей: {e}")
        return {"users": [], "admin": ADMIN_USER_ID}

def save_authorized_users(users_data):
    """Сохраняет список авторизованных пользователей в файл"""
    try:
        with open(USERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(users_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Пользователи успешно сохранены в файл {USERS_FILE}")
        return True
    except Exception as e:
        logger.error(f"Ошибка при сохранении пользователей: {e}")
        return False

def update_user_telegram_id(username: str, user_id: int) -> bool:
    """Обновляет telegram_id пользователя в базе данных"""
    try:
        conn = get_db_connection()
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        # Обновляем user_id в таблице user_folders
        cursor.execute('''
            UPDATE user_folders 
            SET user_id = %s 
            WHERE username = %s AND user_id IS NULL
        ''', (user_id, username))
        
        # Создаем категории и настройки для пользователя, если их еще нет
        if cursor.rowcount > 0:  # Если была обновлена запись
            # Создаем стандартные категории
            default_categories = [
                ("Продукты", ["хлеб", "молоко", "мясо", "овощи", "фрукты"]),
                ("Транспорт", ["бензин", "такси", "автобус", "метро"]),
                ("Развлечения", ["кино", "игры", "кафе", "ресторан"]),
                ("Здоровье", ["лекарства", "врач", "спорт", "аптека"]),
                ("Одежда", ["рубашка", "джинсы", "обувь", "куртка"]),
                ("Коммунальные", ["электричество", "вода", "газ", "интернет"]),
                ("Образование", ["книги", "курсы", "учеба", "тренинги"]),
                ("Прочее", [])
            ]
            
            for category_name, keywords in default_categories:
                cursor.execute('''
                    INSERT INTO user_categories (user_id, category_name, keywords)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (user_id, category_name) DO NOTHING
                ''', (user_id, category_name, keywords))
            
            # Создаем начальные настройки
            initial_settings = [
                ("notifications", {"enabled": True, "email": False, "telegram": True}),
                ("currency", {"code": "Tg", "symbol": "₸"}),
                ("language", {"code": "ru", "name": "Русский"}),
                ("backup", {"auto_backup": True, "frequency": "daily"})
            ]
            
            for setting_key, setting_value in initial_settings:
                cursor.execute('''
                    INSERT INTO user_settings (user_id, setting_key, setting_value)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (user_id, setting_key) DO NOTHING
                ''', (user_id, setting_key, json.dumps(setting_value)))
            
            # Добавляем лог
            cursor.execute('''
                INSERT INTO user_logs (user_id, log_level, message)
                VALUES (%s, %s, %s)
            ''', (user_id, "INFO", f"Пользователь {username} привязан к Telegram ID {user_id}"))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Обновлен telegram_id для пользователя {username}: {user_id}")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при обновлении telegram_id для {username}: {e}")
        return False

def is_user_authorized(user_id: int) -> bool:
    """Проверяет, авторизован ли пользователь"""
    users_data = load_authorized_users()
    
    logger.info(f"Проверка авторизации для user_id: {user_id}")
    logger.info(f"Данные пользователей: {users_data}")
    
    # Проверяем, является ли пользователь админом
    if user_id == users_data.get("admin"):
        logger.info(f"Пользователь {user_id} является админом")
        return True
    
    # Проверяем, есть ли пользователь в списке авторизованных
    for user in users_data.get("users", []):
        if user.get("telegram_id") == user_id:
            logger.info(f"Пользователь {user_id} найден в списке авторизованных")
            return True
    
    logger.info(f"Пользователь {user_id} не авторизован")
    return False

def is_username_authorized(username: str) -> bool:
    """Проверяет, авторизован ли пользователь по имени"""
    users_data = load_authorized_users()
    
    logger.info(f"Проверка username '{username}' в списке авторизованных")
    logger.info(f"Данные пользователей: {users_data}")
    
    # Проверяем, есть ли пользователь с таким именем в списке авторизованных
    for user in users_data.get("users", []):
        if user.get("username") == username:
            logger.info(f"Username '{username}' найден в списке авторизованных")
            return True
    
    logger.info(f"Username '{username}' не найден в списке авторизованных")
    return False

def add_authorized_user(username: str, user_id: int = None, folder_name: str = None, role: str = "user") -> tuple[bool, str]:
    """Добавляет нового авторизованного пользователя с созданием персональной папки"""
    try:
        users_data = load_authorized_users()
        
        # Проверяем, не существует ли уже пользователь с таким именем
        for user in users_data.get("users", []):
            if user.get("username") == username:
                return False, "Пользователь с таким именем уже существует"
        
        # Генерируем уникальное название папки, если не задано
        if not folder_name:
            folder_name = f"user_{username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Добавляем нового пользователя с дополнительными полями
        new_user = {
            "username": username,
            "added_date": datetime.now().isoformat(),
            "status": "active",
            "role": role,
            "folder_name": folder_name
        }
        
        if user_id:
            new_user["telegram_id"] = user_id
        
        users_data["users"].append(new_user)
        
        logger.info(f"Добавлен новый пользователь: {new_user}")
        
        if save_authorized_users(users_data):
            # Создаем персональную папку для пользователя в базе данных
            folder_created, folder_message = create_user_folder(username, folder_name, user_id)
            if folder_created:
                logger.info(f"Персональная папка создана для пользователя {username}")
            else:
                logger.warning(f"Не удалось создать папку для пользователя {username}: {folder_message}")
            
            logger.info(f"Пользователь '{username}' успешно сохранен")
            return True, f"Пользователь успешно добавлен с папкой: {folder_name}"
        else:
            logger.error(f"Ошибка при сохранении пользователя '{username}'")
            return False, "Ошибка при сохранении"
            
    except Exception as e:
        logger.error(f"Ошибка при добавлении пользователя: {e}")
        return False, f"Ошибка: {str(e)}"

def get_authorized_users_list() -> list:
    """Возвращает список всех авторизованных пользователей"""
    users_data = load_authorized_users()
    return users_data.get("users", [])

# --- ФУНКЦИИ СОЗДАНИЯ ПЕРСОНАЛЬНЫХ ПАПОК (Railway/Cloud) ---
def create_user_folder(username: str, folder_name: str, user_id: int) -> tuple[bool, str]:
    """Создает персональную папку пользователя в базе данных (для Railway)"""
    try:
        # Создаем запись пользователя в базе данных
        conn = get_db_connection()
        if not conn:
            return False, "Ошибка подключения к базе данных"
        
        cursor = conn.cursor()
        
        # Таблицы уже созданы в init_db(), просто используем их
        
        # Вставляем данные пользователя
        logger.info(f"Создание записи в user_folders: username={username}, user_id={user_id}, folder_name={folder_name}")
        cursor.execute('''
            INSERT INTO user_folders (username, user_id, folder_name, role, settings, permissions)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (username, user_id) 
            DO UPDATE SET folder_name = EXCLUDED.folder_name
        ''', (
            username, 
            None,  # Всегда NULL при создании, обновится при первом входе
            folder_name, 
            'user',
            json.dumps({
                "currency": "Tg",
                "language": "ru", 
                "notifications": True,
                "auto_classification": True
            }),
            json.dumps({
                "add_expenses": True,
                "view_reports": True,
                "manage_reminders": True,
                "planning": True,
                "analytics": True
            })
        ))
        
        # Создаем стандартные категории для пользователя
        default_categories = [
            ("Продукты", ["хлеб", "молоко", "мясо", "овощи", "фрукты"]),
            ("Транспорт", ["бензин", "такси", "автобус", "метро"]),
            ("Развлечения", ["кино", "игры", "кафе", "ресторан"]),
            ("Здоровье", ["лекарства", "врач", "спорт", "аптека"]),
            ("Одежда", ["рубашка", "джинсы", "обувь", "куртка"]),
            ("Коммунальные", ["электричество", "вода", "газ", "интернет"]),
            ("Образование", ["книги", "курсы", "учеба", "тренинги"]),
            ("Прочее", [])
        ]
        
        # Категории и настройки будут созданы при первом входе пользователя
        # когда user_id будет обновлен на реальный
        
        conn.commit()
        
        # Проверяем, что запись была создана
        cursor.execute('SELECT COUNT(*) FROM user_folders WHERE username = %s', (username,))
        count = cursor.fetchone()[0]
        logger.info(f"Количество записей для пользователя {username}: {count}")
        
        conn.close()
        
        if count > 0:
            logger.info(f"Создана персональная папка в БД для пользователя {username}: {folder_name}")
            return True, f"Персональная папка создана в облаке: {folder_name}"
        else:
            logger.error(f"Не удалось создать запись для пользователя {username}")
            return False, "Ошибка при создании записи в базе данных"
        
    except Exception as e:
        logger.error(f"Ошибка при создании папки пользователя {username}: {e}")
        return False, f"Ошибка при создании папки: {e}"

# Функция create_user_config_files удалена - теперь используется база данных

def get_user_folder_info(username: str, user_id: int) -> dict:
    """Возвращает информацию о папке пользователя из базы данных"""
    try:
        conn = get_db_connection()
        if not conn:
            return None
        
        cursor = conn.cursor()
        cursor.execute('''
            SELECT username, user_id, folder_name, role, settings, permissions, created_at
            FROM user_folders 
            WHERE username = %s AND user_id = %s
        ''', (username, user_id))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                "username": result[0],
                "user_id": result[1],
                "folder_name": result[2],
                "role": result[3],
                "settings": result[4] if result[4] else {},
                "permissions": result[5] if result[5] else {},
                "created_at": result[6]
            }
        return None
        
    except Exception as e:
        logger.error(f"Ошибка при получении информации о папке пользователя {username}: {e}")
        return None

def get_user_config(username: str, user_id: int) -> dict:
    """Загружает конфигурацию пользователя из базы данных"""
    try:
        conn = get_db_connection()
        if not conn:
            return None
        
        cursor = conn.cursor()
        cursor.execute('''
            SELECT settings, permissions, role
            FROM user_folders 
            WHERE username = %s AND user_id = %s
        ''', (username, user_id))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                "settings": result[0] if result[0] else {},
                "permissions": result[1] if result[1] else {},
                "role": result[2] if result[2] else "user"
            }
        return None
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке конфигурации пользователя {username}: {e}")
        return None

# --- СИСТЕМА ИЗОЛЯЦИИ ДАННЫХ (Railway/Cloud) ---
def save_user_data(username: str, user_id: int, data_type: str, data: dict) -> bool:
    """Сохраняет данные пользователя в базе данных"""
    try:
        conn = get_db_connection()
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        # Таблица user_data уже создана в init_db()
        
        # Сохраняем данные
        cursor.execute('''
            INSERT INTO user_data (user_id, data_type, data_content)
            VALUES (%s, %s, %s)
            ON CONFLICT (user_id, data_type) 
            DO UPDATE SET data_content = EXCLUDED.data_content, updated_at = CURRENT_TIMESTAMP
        ''', (user_id, data_type, json.dumps(data)))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Данные {data_type} сохранены для пользователя {username} в БД")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при сохранении данных {data_type} для пользователя {username}: {e}")
        return False

def load_user_data(username: str, user_id: int, data_type: str) -> dict:
    """Загружает данные пользователя из базы данных"""
    try:
        conn = get_db_connection()
        if not conn:
            return {}
        
        cursor = conn.cursor()
        cursor.execute('''
            SELECT data_content FROM user_data 
            WHERE user_id = %s AND data_type = %s
        ''', (user_id, data_type))
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0]:
            return result[0]
        return {}
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных {data_type} для пользователя {username}: {e}")
        return {}

def create_user_backup(username: str, user_id: int) -> bool:
    """Создает резервную копию данных пользователя в базе данных"""
    try:
        conn = get_db_connection()
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        # Таблица user_backups уже создана в init_db()
        
        # Собираем все данные пользователя
        backup_data = {
            "username": username,
            "user_id": user_id,
            "backup_date": datetime.now().isoformat(),
            "data": {
                "expenses": load_user_data(username, user_id, "expenses"),
                "reminders": load_user_data(username, user_id, "reminders"),
                "budget_plans": load_user_data(username, user_id, "budget_plans"),
                "categories": load_user_data(username, user_id, "categories"),
                "settings": load_user_data(username, user_id, "settings")
            }
        }
        
        # Сохраняем резервную копию
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{timestamp}"
        
        cursor.execute('''
            INSERT INTO user_backups (user_id, backup_name, backup_data)
            VALUES (%s, %s, %s)
        ''', (user_id, backup_name, json.dumps(backup_data)))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Резервная копия создана для пользователя {username} в БД: {backup_name}")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при создании резервной копии для пользователя {username}: {e}")
        return False

def restore_user_backup(username: str, user_id: int, backup_id: int) -> bool:
    """Восстанавливает данные пользователя из резервной копии в базе данных"""
    try:
        conn = get_db_connection()
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        # Получаем резервную копию
        cursor.execute('''
            SELECT backup_data FROM user_backups 
            WHERE id = %s AND user_id = %s
        ''', (backup_id, user_id))
        
        result = cursor.fetchone()
        if not result:
            return False
        
        backup_data = result[0]
        
        # Восстанавливаем данные
        for data_type, data in backup_data.get("data", {}).items():
            save_user_data(username, user_id, data_type, data)
        
        conn.close()
        
        logger.info(f"Данные восстановлены для пользователя {username} из резервной копии {backup_id}")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при восстановлении данных для пользователя {username}: {e}")
        return False

def log_user_action(user_id: int, action: str, details: str = "") -> None:
    """Логирует действие пользователя в базу данных"""
    try:
        conn = get_db_connection()
        if not conn:
            return
        
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO user_logs (user_id, log_level, message)
            VALUES (%s, %s, %s)
        ''', (user_id, "INFO", f"{action}: {details}"))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logger.error(f"Ошибка при логировании действия пользователя {user_id}: {e}")

# --- ФУНКЦИИ УПРАВЛЕНИЯ ГРУППАМИ ---
def create_group(name: str, admin_user_id: int) -> tuple[bool, str, str]:
    """Создает новую группу и возвращает (успех, сообщение, код приглашения)"""
    try:
        conn = get_db_connection()
        if not conn:
            return False, "Ошибка подключения к базе данных", ""
        
        cursor = conn.cursor()
        
        # Генерируем уникальный код приглашения
        import secrets
        import string
        invitation_code = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(8))
        
        # Создаем группу
        cursor.execute('''
            INSERT INTO groups (name, admin_user_id, invitation_code)
            VALUES (%s, %s, %s)
            RETURNING id
        ''', (name, admin_user_id, invitation_code))
        
        group_id = cursor.fetchone()[0]
        
        # Добавляем админа как участника группы
        cursor.execute('''
            INSERT INTO group_members (group_id, user_id, phone, role)
            VALUES (%s, %s, %s, %s)
        ''', (group_id, admin_user_id, "admin", "admin"))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Группа '{name}' создана с ID {group_id}, пользователь {admin_user_id} добавлен как админ")
        
        return True, f"Группа '{name}' успешно создана", invitation_code
        
    except Exception as e:
        logger.error(f"Ошибка при создании группы: {e}")
        if conn:
            conn.rollback()
            conn.close()
        return False, f"Ошибка при создании группы: {str(e)}", ""

def get_user_group(user_id: int) -> dict:
    """Получает информацию о группе пользователя"""
    try:
        conn = get_db_connection()
        if not conn:
            return None
        
        cursor = conn.cursor()
        cursor.execute('''
            SELECT g.id, g.name, g.admin_user_id, g.invitation_code, gm.role
            FROM groups g
            JOIN group_members gm ON g.id = gm.group_id
            WHERE gm.user_id = %s
        ''', (user_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            group_info = {
                "id": result[0],
                "name": result[1],
                "admin_user_id": result[2],
                "invitation_code": result[3],
                "role": result[4]
            }
            logger.info(f"Найдена группа для пользователя {user_id}: {group_info}")
            return group_info
        
        logger.info(f"Группа для пользователя {user_id} не найдена")
        return None
        
    except Exception as e:
        logger.error(f"Ошибка при получении группы пользователя: {e}")
        return None

def join_group_by_invitation(invitation_code: str, user_id: int, phone: str) -> tuple[bool, str]:
    """Присоединяет пользователя к группе по коду приглашения"""
    try:
        conn = get_db_connection()
        if not conn:
            return False, "Ошибка подключения к базе данных"
        
        cursor = conn.cursor()
        
        # Проверяем код приглашения
        cursor.execute('''
            SELECT id, name
            FROM groups
            WHERE invitation_code = %s
        ''', (invitation_code,))
        
        group_info = cursor.fetchone()
        if not group_info:
            return False, "Неверный код приглашения"
        
        group_id, group_name = group_info
        
        # Проверяем количество участников (максимум 5)
        cursor.execute('''
            SELECT COUNT(*) FROM group_members WHERE group_id = %s
        ''', (group_id,))
        
        current_members = cursor.fetchone()[0]
        if current_members >= 5:
            return False, f"Группа '{group_name}' уже заполнена (максимум 5 участников)"
        
        # Проверяем, не является ли пользователь уже участником
        cursor.execute('''
            SELECT id FROM group_members 
            WHERE group_id = %s AND user_id = %s
        ''', (group_id, user_id))
        
        if cursor.fetchone():
            return False, "Вы уже являетесь участником этой группы"
        
        # Добавляем пользователя в группу
        cursor.execute('''
            INSERT INTO group_members (group_id, user_id, phone, role)
            VALUES (%s, %s, %s, %s)
        ''', (group_id, user_id, phone, "member"))
        
        conn.commit()
        conn.close()
        
        return True, f"Вы успешно присоединились к группе '{group_name}'"
        
    except Exception as e:
        logger.error(f"Ошибка при присоединении к группе: {e}")
        if conn:
            conn.rollback()
            conn.close()
        return False, f"Ошибка при присоединении к группе: {str(e)}"

def is_user_in_group(user_id: int) -> bool:
    """Проверяет, находится ли пользователь в какой-либо группе"""
    result = get_user_group(user_id) is not None
    logger.info(f"Пользователь {user_id} в группе: {result}")
    return result

def get_group_members(group_id: int) -> list:
    """Получает список участников группы"""
    try:
        conn = get_db_connection()
        if not conn:
            return []
        
        cursor = conn.cursor()
        cursor.execute('''
            SELECT gm.user_id, gm.phone, gm.role, gm.joined_at
            FROM group_members gm
            WHERE gm.group_id = %s
            ORDER BY gm.joined_at
        ''', (group_id,))
        
        members = []
        for row in cursor.fetchall():
            members.append({
                "user_id": row[0],
                "phone": row[1],
                "role": row[2],
                "joined_at": row[3]
            })
        
        conn.close()
        return members
        
    except Exception as e:
        logger.error(f"Ошибка при получении участников группы: {e}")
        return []

# Обновляем функцию проверки доступа
def validate_block_access(block_name: str, user_id: int) -> bool:
    """Проверяет доступ пользователя к блоку"""
    if not is_block_protected(block_name):
        return False
    
    # Проверяем авторизацию пользователя
    if not is_user_authorized(user_id):
        return False
    
    return True

if __name__ == "__main__":
    main()