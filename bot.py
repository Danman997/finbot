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
import numpy as np
import pandas as pd
import re
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
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
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
        
        conn.commit()
        conn.close()
        logger.info("База данных инициализирована (таблицы 'expenses' и 'payment_reminders' проверены/созданы).")

def add_expense(amount, category, description, transaction_date):
    conn = get_db_connection()
    if not conn:
        return False
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO expenses (amount, category, description, transaction_date)
            VALUES (%s, %s, %s, %s)
        ''', (amount, category, description, transaction_date))
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

# --- UI (User Interface) ---
def get_main_menu_keyboard():
    keyboard = [
        [KeyboardButton("💸 Добавить расход"), KeyboardButton("📊 Отчеты")],
        [KeyboardButton("🔧 Исправить категории"), KeyboardButton("📚 Обучить модель")],
        [KeyboardButton("⏰ Напоминания"), KeyboardButton("📅 Планирование")],
        [KeyboardButton("📈 Аналитика")]
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
    await update.message.reply_text(
        "Привет! Я твой помощник по учету расходов. Выбери опцию ниже:",
        reply_markup=get_main_menu_keyboard()
    )

async def report_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "За какой период вы хотите отчет?",
        reply_markup=get_report_period_keyboard()
    )
    return PERIOD_CHOICE_STATE  # Важно! Переводим в состояние выбора периода

async def correction_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Меню исправления категорий"""
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
        
        await update.message.reply_text(
            reminders_text,
            reply_markup=ReplyKeyboardMarkup([["🔙 Назад"]], resize_keyboard=True)
        )
        return REMINDER_MENU_STATE
    
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
        if add_expense(amount, category, description, transaction_date): 
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

# --- Доп. состояния для планирования бюджета ---
PLAN_MENU_STATE = 22
PLAN_MONTH_STATE = 23
PLAN_TOTAL_STATE = 24
PLAN_CATEGORY_STATE = 25
PLAN_AMOUNT_STATE = 26
PLAN_COMMENT_STATE = 27
PLAN_SUMMARY_STATE = 28
PLAN_DELETE_STATE = 29

# --- Состояния для аналитики ---
ANALYTICS_MENU_STATE = 30
ANALYTICS_MONTH_STATE = 31
ANALYTICS_REPORT_STATE = 32

# --- Функции для аналитики ---
async def analytics_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Меню аналитики"""
    keyboard = [
        [KeyboardButton("📊 Анализ месяца")],
        [KeyboardButton("📈 Сравнение планов")],
        [KeyboardButton("🔙 Назад")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)
    
    await update.message.reply_text(
        "📈 Аналитика и финансовые рекомендации\n\n"
        "📊 Анализ месяца - сравнение расходов с планированием\n"
        "📈 Сравнение планов - анализ трендов и рекомендации\n"
        "🔙 Назад - вернуться в главное меню",
        reply_markup=reply_markup
    )
    return ANALYTICS_MENU_STATE

async def analytics_month_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Выбор месяца для анализа"""
    text = update.message.text.strip()
    
    if text == "📊 Анализ месяца":
        # Показываем доступные месяцы для анализа
        months = get_available_months_for_analytics()
        if not months:
            await update.message.reply_text(
                "❌ Нет данных для анализа. Сначала создайте планирование и добавьте расходы.",
                reply_markup=get_main_menu_keyboard()
            )
            return ConversationHandler.END
        
        keyboard = []
        for month in months:
            keyboard.append([KeyboardButton(month)])
        keyboard.append([KeyboardButton("🔙 Назад")])
        
        reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)
        
        await update.message.reply_text(
            "📊 Выберите месяц для анализа:\n\n"
            "Анализ покажет:\n"
            "• Сравнение плана vs факт\n"
            "• Превышения бюджета\n"
            "• Рекомендации по экономии",
            reply_markup=reply_markup
        )
        return ANALYTICS_MONTH_STATE
        
    elif text == "📈 Сравнение планов":
        await generate_comparison_analytics(update, context)
        return ConversationHandler.END
        
    elif text == "🔙 Назад":
        await update.message.reply_text(
            "Возвращаемся в главное меню",
            reply_markup=get_main_menu_keyboard()
        )
        return ConversationHandler.END
    
    return ANALYTICS_MENU_STATE

async def analytics_month_selected(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбранного месяца для анализа"""
    text = update.message.text.strip()
    
    if text == "🔙 Назад":
        await analytics_menu(update, context)
        return ANALYTICS_MENU_STATE
    
    # Парсим месяц и год
    try:
        month, year = parse_month_year(text)
        await generate_monthly_analytics(update, context, month, year)
        return ConversationHandler.END
    except ValueError:
        await update.message.reply_text(
            "❌ Неверный формат месяца. Попробуйте снова.",
            reply_markup=get_main_menu_keyboard()
        )
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

async def generate_monthly_analytics(update: Update, context: ContextTypes.DEFAULT_TYPE, month: int, year: int):
    """Генерация аналитики за месяц"""
    try:
        # Получаем план на месяц
        plan = get_budget_plan_by_month(month, year)
        if not plan:
            await update.message.reply_text(
                "❌ План на выбранный месяц не найден.",
                reply_markup=get_main_menu_keyboard()
            )
            return
        
        plan_id, total_budget = plan
        
        # Получаем фактические расходы за месяц
        expenses = get_monthly_expenses(month, year)
        
        # Получаем детали плана
        plan_items = get_budget_plan_items(plan_id)
        
        # Анализируем данные
        analysis_result = analyze_monthly_data(plan_items, expenses, total_budget)
        
        # Генерируем график сравнения
        fig = create_analytics_chart(plan_items, expenses, total_budget)
        
        # Отправляем результат
        await update.message.reply_photo(
            photo=fig,
            caption=analysis_result,
            reply_markup=get_main_menu_keyboard()
        )
        
        plt.close(fig)
        
    except Exception as e:
        logger.error(f"Ошибка при генерации аналитики: {e}")
        await update.message.reply_text(
            f"❌ Ошибка при генерации аналитики: {e}",
            reply_markup=get_main_menu_keyboard()
        )

async def generate_comparison_analytics(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Генерация сравнительной аналитики"""
    try:
        # Получаем данные для сравнения
        comparison_data = get_comparison_data()
        
        if not comparison_data:
            await update.message.reply_text(
                "❌ Недостаточно данных для сравнения. Нужны данные за несколько месяцев.",
                reply_markup=get_main_menu_keyboard()
            )
            return
        
        # Генерируем график сравнения
        fig = create_comparison_chart(comparison_data)
        
        # Формируем рекомендации
        recommendations = generate_financial_recommendations(comparison_data)
        
        await update.message.reply_photo(
            photo=fig,
            caption=recommendations,
            reply_markup=get_main_menu_keyboard()
        )
        
        plt.close(fig)
        
    except Exception as e:
        logger.error(f"Ошибка при генерации сравнительной аналитики: {e}")
        await update.message.reply_text(
            f"❌ Ошибка при генерации сравнительной аналитики: {e}",
            reply_markup=get_main_menu_keyboard()
        )

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
        
        return cursor.fetchall()
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
        
        return cursor.fetchall()
    except Exception as e:
        logger.error(f"Ошибка при получении элементов плана: {e}")
        return []
    finally:
        conn.close()

def analyze_monthly_data(plan_items, expenses, total_budget):
    """Анализ месячных данных"""
    # Создаем словари для удобства
    plan_dict = {item[0]: item[1] for item in plan_items}
    expense_dict = {item[0]: item[1] for item in expenses}
    
    # Анализируем каждую категорию
    analysis = []
    total_planned = sum(plan_dict.values())
    total_spent = sum(expense_dict.values())
    
    # Сравниваем план vs факт
    for category in set(plan_dict.keys()) | set(expense_dict.keys()):
        planned = plan_dict.get(category, 0)
        spent = expense_dict.get(category, 0)
        
        if spent > planned:
            overspend = spent - planned
            overspend_percent = (overspend / planned * 100) if planned > 0 else 0
            analysis.append(f"🔴 {category}: превышение на {overspend:.0f} Тг ({overspend_percent:.1f}%)")
        elif spent < planned:
            saved = planned - spent
            saved_percent = (saved / planned * 100) if planned > 0 else 0
            analysis.append(f"🟢 {category}: сэкономлено {saved:.0f} Тг ({saved_percent:.1f}%)")
        else:
            analysis.append(f"🟡 {category}: точно по плану")
    
    # Общий анализ
    overall_analysis = f"📊 ОБЩИЙ АНАЛИЗ МЕСЯЦА\n\n"
    overall_analysis += f"💰 Планируемый бюджет: {total_budget:,.0f} Тг\n"
    overall_analysis += f"💸 Фактические расходы: {total_spent:,.0f} Тг\n"
    
    if total_spent > total_budget:
        overspend_total = total_spent - total_budget
        overall_analysis += f"🔴 Превышение бюджета: {overspend_total:,.0f} Тг\n"
    else:
        saved_total = total_budget - total_spent
        overall_analysis += f"🟢 Сэкономлено: {saved_total:,.0f} Тг\n"
    
    overall_analysis += f"\n📋 ДЕТАЛЬНЫЙ АНАЛИЗ ПО КАТЕГОРИЯМ:\n"
    overall_analysis += "\n".join(analysis)
    
    # Рекомендации
    recommendations = generate_category_recommendations(plan_dict, expense_dict)
    overall_analysis += f"\n\n💡 РЕКОМЕНДАЦИИ:\n{recommendations}"
    
    return overall_analysis

def generate_category_recommendations(plan_dict, expense_dict):
    """Генерация рекомендаций по категориям"""
    recommendations = []
    
    for category in set(plan_dict.keys()) | set(expense_dict.keys()):
        planned = plan_dict.get(category, 0)
        spent = expense_dict.get(category, 0)
        
        if spent > planned * 1.2:  # Превышение более чем на 20%
            recommendations.append(f"• {category}: Рассмотрите увеличение бюджета на 15-20%")
        elif spent < planned * 0.8:  # Экономия более чем на 20%
            recommendations.append(f"• {category}: Возможно, бюджет можно сократить на 10-15%")
    
    if not recommendations:
        recommendations.append("• Ваш бюджет хорошо сбалансирован!")
    
    return "\n".join(recommendations)

def create_analytics_chart(plan_items, expenses, total_budget):
    """Создание графика аналитики"""
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#1a1a1a')
    
    # Цветовая палитра
    colors = ['#6B8E23', '#4682B4', '#CD853F', '#20B2AA', '#8A2BE2', '#32CD32', '#FF8C00', '#DC143C', '#1E90FF', '#9370DB']
    
    # Создаем данные для сравнения
    categories = []
    planned_amounts = []
    actual_amounts = []
    
    all_categories = set(item[0] for item in plan_items) | set(item[0] for item in expenses)
    
    for category in all_categories:
        categories.append(category)
        planned = next((item[1] for item in plan_items if item[0] == category), 0)
        actual = next((item[1] for item in expenses if item[0] == category), 0)
        planned_amounts.append(planned)
        actual_amounts.append(actual)
    
    # График сравнения план vs факт
    ax1 = fig.add_subplot(2, 2, 1)
    x = range(len(categories))
    width = 0.35
    
    bars1 = ax1.bar([i - width/2 for i in x], planned_amounts, width, label='План', color='#4682B4', alpha=0.8)
    bars2 = ax1.bar([i + width/2 for i in x], actual_amounts, width, label='Факт', color='#CD853F', alpha=0.8)
    
    ax1.set_title('ПЛАН VS ФАКТ ПО КАТЕГОРИЯМ', color='white', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Сумма (Тг)', color='white', fontsize=14)
    ax1.set_xlabel('Категории', color='white', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha='right', color='white')
    ax1.tick_params(colors='white')
    ax1.legend()
    ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    # Круговая диаграмма распределения бюджета
    ax2 = fig.add_subplot(2, 2, 2)
    wedges, texts, autotexts = ax2.pie(planned_amounts, labels=categories, autopct='%1.1f%%',
                                       startangle=90, colors=colors[:len(planned_amounts)],
                                       shadow=True, explode=[0.05] * len(planned_amounts))
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax2.set_title('ПЛАНИРУЕМОЕ РАСПРЕДЕЛЕНИЕ', color='white', fontsize=16, fontweight='bold')
    
    # График превышений/экономии
    ax3 = fig.add_subplot(2, 2, 3)
    differences = [actual - planned for planned, actual in zip(planned_amounts, actual_amounts)]
    colors_diff = ['red' if diff > 0 else 'green' for diff in differences]
    
    bars3 = ax3.bar(categories, differences, color=colors_diff, alpha=0.8)
    ax3.set_title('ПРЕВЫШЕНИЕ/ЭКОНОМИЯ', color='white', fontsize=16, fontweight='bold')
    ax3.set_ylabel('Разница (Тг)', color='white', fontsize=14)
    ax3.set_xticklabels(categories, rotation=45, ha='right', color='white')
    ax3.tick_params(colors='white')
    ax3.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    # Добавляем значения на столбцы
    for bar, diff in zip(bars3, differences):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + (max(differences) if max(differences) > 0 else 0)*0.01,
                 f'{diff:+.0f}', ha='center', va='bottom', color='white', fontweight='bold')
    
    # Общая статистика
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    total_planned = sum(planned_amounts)
    total_actual = sum(actual_amounts)
    
    stats_text = f"📊 ОБЩАЯ СТАТИСТИКА\n\n"
    stats_text += f"💰 Планируемый бюджет: {total_planned:,.0f} Тг\n"
    stats_text += f"💸 Фактические расходы: {total_actual:,.0f} Тг\n"
    
    if total_actual > total_planned:
        overspend = total_actual - total_planned
        stats_text += f"🔴 Превышение: {overspend:,.0f} Тг\n"
    else:
        saved = total_planned - total_actual
        stats_text += f"🟢 Сэкономлено: {saved:,.0f} Тг\n"
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=14, 
             color='white', fontweight='bold', verticalalignment='top')
    
    fig.suptitle('АНАЛИТИКА РАСХОДОВ VS ПЛАНИРОВАНИЕ', color='white', fontsize=20, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Сохраняем в BytesIO
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', facecolor='#1a1a1a', bbox_inches='tight', dpi=300)
    img_buffer.seek(0)
    
    return img_buffer

def get_comparison_data():
    """Получить данные для сравнения"""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT 
                EXTRACT(MONTH FROM bp.plan_month) as month,
                EXTRACT(YEAR FROM bp.plan_month) as year,
                bp.total_amount,
                SUM(e.amount) as total_expenses
            FROM budget_plans bp
            LEFT JOIN expenses e ON 
                EXTRACT(MONTH FROM e.transaction_date) = EXTRACT(MONTH FROM bp.plan_month)
                AND EXTRACT(YEAR FROM e.transaction_date) = EXTRACT(YEAR FROM bp.plan_month)
            GROUP BY bp.id, bp.total_amount, bp.plan_month
            ORDER BY bp.plan_month DESC
            LIMIT 6
        ''')
        
        return cursor.fetchall()
    except Exception as e:
        logger.error(f"Ошибка при получении данных для сравнения: {e}")
        return []
    finally:
        conn.close()

def create_comparison_chart(comparison_data):
    """Создание графика сравнения"""
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#1a1a1a')
    
    # Цветовая палитра
    colors = ['#6B8E23', '#4682B4', '#CD853F', '#20B2AA', '#8A2BE2', '#32CD32']
    
    months = []
    budgets = []
    expenses = []
    
    for row in comparison_data:
        month_num, year = int(row[0]), int(row[1])
        month_name = get_month_name(month_num)
        months.append(f"{month_name} {year}")
        budgets.append(row[2])  # total_amount
        expenses.append(row[3] if row[3] else 0)  # total_expenses
    
    # График сравнения бюджетов и расходов
    ax1 = fig.add_subplot(2, 1, 1)
    x = range(len(months))
    width = 0.35
    
    bars1 = ax1.bar([i - width/2 for i in x], budgets, width, label='Планируемый бюджет', color='#4682B4', alpha=0.8)
    bars2 = ax1.bar([i + width/2 for i in x], expenses, width, label='Фактические расходы', color='#CD853F', alpha=0.8)
    
    ax1.set_title('СРАВНЕНИЕ БЮДЖЕТОВ И РАСХОДОВ ПО МЕСЯЦАМ', color='white', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Сумма (Тг)', color='white', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(months, rotation=45, ha='right', color='white')
    ax1.tick_params(colors='white')
    ax1.legend()
    ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    # График трендов
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(months, budgets, 'o-', label='Планируемый бюджет', color='#4682B4', linewidth=3, markersize=8)
    ax2.plot(months, expenses, 's-', label='Фактические расходы', color='#CD853F', linewidth=3, markersize=8)
    
    ax2.set_title('ТРЕНДЫ БЮДЖЕТИРОВАНИЯ', color='white', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Сумма (Тг)', color='white', fontsize=14)
    ax2.set_xticklabels(months, rotation=45, ha='right', color='white')
    ax2.tick_params(colors='white')
    ax2.legend()
    ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    fig.suptitle('СРАВНИТЕЛЬНАЯ АНАЛИТИКА ПО МЕСЯЦАМ', color='white', fontsize=20, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Сохраняем в BytesIO
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', facecolor='#1a1a1a', bbox_inches='tight', dpi=300)
    img_buffer.seek(0)
    
    return img_buffer

def generate_financial_recommendations(comparison_data):
    """Генерация финансовых рекомендаций"""
    if not comparison_data:
        return "❌ Недостаточно данных для анализа"
    
    recommendations = "💡 ФИНАНСОВЫЕ РЕКОМЕНДАЦИИ\n\n"
    
    # Анализируем тренды
    total_budget = sum(row[2] for row in comparison_data)  # total_amount
    total_expenses = sum(row[3] if row[3] else 0 for row in comparison_data)  # total_expenses
    
    if total_expenses > total_budget:
        overspend = total_expenses - total_budget
        overspend_percent = (overspend / total_budget * 100)
        recommendations += f"🔴 Общее превышение бюджета: {overspend:,.0f} Тг ({overspend_percent:.1f}%)\n\n"
        recommendations += "📋 Рекомендации по экономии:\n"
        recommendations += "• Пересмотрите планирование на следующий месяц\n"
        recommendations += "• Увеличьте бюджет на 10-15% для непредвиденных расходов\n"
        recommendations += "• Анализируйте категории с регулярными превышениями\n"
    else:
        saved = total_budget - total_expenses
        saved_percent = (saved / total_budget * 100)
        recommendations += f"🟢 Общая экономия: {saved:,.0f} Тг ({saved_percent:.1f}%)\n\n"
        recommendations += "📋 Рекомендации по оптимизации:\n"
        recommendations += "• Ваш бюджет хорошо сбалансирован\n"
        recommendations += "• Рассмотрите увеличение сбережений\n"
        recommendations += "• Возможно, можно сократить некоторые категории\n"
    
    # Анализ по месяцам
    recommendations += "\n📊 АНАЛИЗ ПО МЕСЯЦАМ:\n"
    for row in comparison_data:
        month_num, year = int(row[0]), int(row[1])
        month_name = get_month_name(month_num)
        budget = row[2]  # total_amount
        expenses = row[3] if row[3] else 0  # total_expenses
        
        if expenses > budget:
            overspend = expenses - budget
            recommendations += f"• {month_name} {year}: превышение на {overspend:,.0f} Тг\n"
        else:
            saved = budget - expenses
            recommendations += f"• {month_name} {year}: экономия {saved:,.0f} Тг\n"
    
    return recommendations

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
        fallbacks=[MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)],
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
        fallbacks=[MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)],
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
                MessageHandler(filters.Regex("^(📝 Добавить напоминание|📋 Список напоминаний|🗑️ Удалить напоминание|🔙 Назад)$"), reminder_menu),
                MessageHandler(filters.Regex("^⏰ Напоминания$"), reminder_menu)
            ],
            REMINDER_TITLE_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, reminder_title_input)],
            REMINDER_DESC_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, reminder_desc_input)],
            REMINDER_AMOUNT_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, reminder_amount_input)],
            REMINDER_START_DATE_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, reminder_start_date_input)],
            REMINDER_END_DATE_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, reminder_end_date_input)],
            REMINDER_MANAGE_STATE: [MessageHandler(filters.Regex("^(❌ Удалить \d+|🔙 Назад)$"), reminder_manage)],
            REMINDER_DELETE_STATE: [MessageHandler(filters.Regex("^(❌ Удалить \d+|🔙 Назад)$"), reminder_delete_confirm)],
        },
        fallbacks=[MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)],
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
                MessageHandler(filters.Regex("^(➕ Добавить планирование|📋 Список планов|🗑️ Удалить план|🔙 Назад)$"), planning_menu),
                MessageHandler(filters.Regex("^📅 Планирование$"), planning_menu),
                MessageHandler(filters.Regex("^.* — .*$"), planning_menu)  # Обработка выбора месяца
            ],
            PLAN_MONTH_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, planning_month)],
            PLAN_TOTAL_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, planning_total)],
            PLAN_CATEGORY_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, planning_category)],
            PLAN_AMOUNT_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, planning_amount)],
            PLAN_COMMENT_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, planning_comment)],
            PLAN_DELETE_STATE: [MessageHandler(filters.Regex("^(❌ Удалить план \d+|🔙 Назад)$"), planning_delete_confirm)],
            CUSTOM_CATEGORY_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, custom_category_input)],
        },
        fallbacks=[MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)],
        allow_reentry=True
    )
    
    # Обработчик для аналитики
    analytics_conv_handler = ConversationHandler(
        entry_points=[
            MessageHandler(filters.Regex("^📈 Аналитика$"), analytics_menu),
            CommandHandler("analytics", analytics_menu)
        ],
        states={
            ANALYTICS_MENU_STATE: [
                MessageHandler(filters.Regex("^(📊 Анализ месяца|📈 Сравнение планов|🔙 Назад)$"), analytics_month_choice),
                MessageHandler(filters.Regex("^📈 Аналитика$"), analytics_menu)
            ],
            ANALYTICS_MONTH_STATE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, analytics_month_selected)
            ],
        },
        fallbacks=[MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)],
        allow_reentry=True
    )

    application.add_handler(report_conv_handler)
    application.add_handler(correction_conv_handler)
    application.add_handler(reminder_conv_handler)
    application.add_handler(planning_conv_handler)
    application.add_handler(analytics_conv_handler)
    application.add_handler(CommandHandler("start", start))
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
			return (row[0], float(row[1]))
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
        kb.append([KeyboardButton("🔙 Назад")])
        await update.message.reply_text("\n".join(text_lines), reply_markup=ReplyKeyboardMarkup(kb, resize_keyboard=True))
        # Сохраняем планы в контексте для последующего выбора
        context.user_data['plans_list'] = rows
        return PLAN_MENU_STATE
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

if __name__ == "__main__":
    main()