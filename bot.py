import os
import logging
import psycopg2
from psycopg2 import sql
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
from telegram.ext import Application, CommandHandler, MessageHandler, ConversationHandler, ContextTypes, filters
import matplotlib.pyplot as plt
import io
import re
import schedule
import time

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
        [KeyboardButton("⏰ Напоминания")]
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
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)

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
    new_category = update.message.text
    
    if new_category not in CATEGORIES:
        await update.message.reply_text(
            "Пожалуйста, выберите категорию из предложенных.",
            reply_markup=get_categories_keyboard()
        )
        return CATEGORY_CHOICE_STATE
    
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
            "Категория обновлена. Теперь введите новую сумму (например: 1500.50) или отправьте текущую сумму без изменений:",
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
    
    if text == "📝 Добавить напоминание":
        await update.message.reply_text(
            "Введите информацию о напоминании в формате:\n\n"
            "Название | Описание | Сумма | Дата начала | Дата окончания\n\n"
            "Например:\n"
            "Автострахование | ОСАГО на машину | 25000 | 20.08.2025 | 19.08.2026\n\n"
            "Или просто отправьте название:",
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
        keyboard = []
        for i in range(0, len(reminders), 2):
            row = [KeyboardButton(f"❌ Удалить {i+1}")]
            if i + 1 < len(reminders):
                row.append(KeyboardButton(f"❌ Удалить {i+2}"))
            keyboard.append(row)
        keyboard.append([KeyboardButton("🔙 Назад")])
        
        await update.message.reply_text(
            reminders_text,
            reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
        )
        context.user_data['reminders_list'] = reminders
        return REMINDER_MANAGE_STATE
    
    elif text == "🔙 Назад":
        await update.message.reply_text(
            "Возвращаюсь в главное меню:",
            reply_markup=get_main_menu_keyboard()
        )
        return ConversationHandler.END
    
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
        
        # Группировка данных для различных графиков
        grouped_by_category = df.groupby('Категория', as_index=False)['Сумма'].sum().sort_values(by='Сумма', ascending=False)
        grouped_by_month = df.groupby('Месяц', as_index=False)['Сумма'].sum()
        grouped_by_weekday = df.groupby('День недели', as_index=False)['Сумма'].sum()
        
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

    # Создание комплексного дашборда
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor('#1a1a1a')  # Темный фон
    
    # Настройка стиля
    plt.style.use('dark_background')
    
    # Цветовая палитра
    colors = ['#00ff88', '#00d4ff', '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3', '#54a0ff', '#5f27cd']
    
    # Верхний ряд - KPI метрики
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # 1. Общие расходы (полукруговая диаграмма)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.pie([total, max_expense * 10], labels=['', ''], colors=['#00ff88', '#1a1a1a'], 
             startangle=90, counterclock=False, radius=0.8)
    ax1.text(0, 0, f'Общие\nрасходы\n{total:.0f} Тг', ha='center', va='center', 
              fontsize=14, fontweight='bold', color='white')
    ax1.set_title('ОБЩИЕ РАСХОДЫ', color='white', fontsize=12, fontweight='bold', pad=20)
    
    # 2. Средний расход
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.pie([avg_expense, max_expense * 5], labels=['', ''], colors=['#00d4ff', '#1a1a1a'], 
             startangle=90, counterclock=False, radius=0.8)
    ax2.text(0, 0, f'Средний\nрасход\n{avg_expense:.0f} Тг', ha='center', va='center', 
              fontsize=14, fontweight='bold', color='white')
    ax2.set_title('СРЕДНИЙ РАСХОД', color='white', fontsize=12, fontweight='bold', pad=20)
    
    # 3. Количество транзакций
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.pie([total_transactions, 100], labels=['', ''], colors=['#ff6b6b', '#1a1a1a'], 
             startangle=90, counterclock=False, radius=0.8)
    ax3.text(0, 0, f'Транзакции\n{total_transactions}', ha='center', va='center', 
              fontsize=14, fontweight='bold', color='white')
    ax3.set_title('ТРАНЗАКЦИИ', color='white', fontsize=12, fontweight='bold', pad=20)
    
    # 4. Максимальный расход
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.pie([max_expense, max_expense * 2], labels=['', ''], colors=['#4ecdc4', '#1a1a1a'], 
             startangle=90, counterclock=False, radius=0.8)
    ax4.text(0, 0, f'Макс.\nрасход\n{max_expense:.0f} Тг', ha='center', va='center', 
              fontsize=14, fontweight='bold', color='white')
    ax4.set_title('МАКСИМАЛЬНЫЙ РАСХОД', color='white', fontsize=12, fontweight='bold', pad=20)
    
    # Средний ряд - Анализ по месяцам и категориям
    
    # 5. Расходы по месяцам (столбчатая диаграмма)
    ax5 = fig.add_subplot(gs[1, :2])
    months = grouped_by_month['Месяц'].tolist()
    month_amounts = grouped_by_month['Сумма'].tolist()
    
    bars = ax5.bar(months, month_amounts, color=colors[:len(months)], alpha=0.8)
    ax5.set_title('РАСХОДЫ ПО МЕСЯЦАМ', color='white', fontsize=14, fontweight='bold', pad=20)
    ax5.set_ylabel('Сумма (Тг)', color='white', fontsize=12)
    ax5.set_xlabel('Месяц', color='white', fontsize=12)
    ax5.tick_params(colors='white')
    ax5.grid(True, alpha=0.3)
    
    # Добавляем значения на столбцы
    for bar, amount in zip(bars, month_amounts):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + max(month_amounts)*0.01,
                 f'{amount:.0f}', ha='center', va='bottom', color='white', fontweight='bold')
    
    # 6. Анализ по категориям (кольцевая диаграмма)
    ax6 = fig.add_subplot(gs[1, 2:])
    wedges, texts, autotexts = ax6.pie(amounts, labels=None, autopct='%1.1f%%', 
                                        startangle=90, pctdistance=0.85, colors=colors[:len(amounts)],
                                        wedgeprops=dict(width=0.5))
    ax6.set_title('АНАЛИЗ ПО КАТЕГОРИЯМ\nРАСХОДЫ ПО КАТЕГОРИЯМ', color='white', fontsize=14, fontweight='bold', pad=20)
    
    # Центральный текст
    ax6.text(0, 0, f'РАСХОДЫ\n{total:.0f} Тг', ha='center', va='center', 
              fontsize=16, fontweight='bold', color='white')
    
    # Легенда справа
    legend_labels = [f"{cat} — {amt:.0f} Тг" for cat, amt in zip(categories, amounts)]
    ax6.legend(wedges, legend_labels, title="Категории", loc="center left", 
               bbox_to_anchor=(1.1, 0.5), fontsize=10, title_fontsize=12)
    
    # Нижний ряд - Дополнительная аналитика
    
    # 7. Расходы по дням недели
    ax7 = fig.add_subplot(gs[2, :2])
    weekdays = grouped_by_weekday['День недели'].tolist()
    weekday_amounts = grouped_by_weekday['Сумма'].tolist()
    
    bars = ax7.bar(weekdays, weekday_amounts, color=colors[5:5+len(weekdays)], alpha=0.8)
    ax7.set_title('РАСХОДЫ ПО ДНЯМ НЕДЕЛИ', color='white', fontsize=14, fontweight='bold', pad=20)
    ax7.set_ylabel('Сумма (Тг)', color='white', fontsize=12)
    ax7.set_xlabel('День недели', color='white', fontsize=12)
    ax7.tick_params(colors='white')
    ax7.grid(True, alpha=0.3)
    
    # Добавляем значения на столбцы
    for bar, amount in zip(bars, weekday_amounts):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + max(weekday_amounts)*0.01,
                 f'{amount:.0f}', ha='center', va='bottom', color='white', fontweight='bold')
    
    # 8. Топ-5 категорий (горизонтальная столбчатая диаграмма)
    ax8 = fig.add_subplot(gs[2, 2:])
    top_categories = categories[:5]
    top_amounts = amounts[:5]
    
    bars = ax8.barh(top_categories, top_amounts, color=colors[:5], alpha=0.8)
    ax8.set_title('ТОП-5 КАТЕГОРИЙ', color='white', fontsize=14, fontweight='bold', pad=20)
    ax8.set_xlabel('Сумма (Тг)', color='white', fontsize=12)
    ax8.tick_params(colors='white')
    ax8.grid(True, alpha=0.3)
    
    # Добавляем значения на столбцы
    for bar, amount in zip(bars, top_amounts):
        width = bar.get_width()
        ax8.text(width + max(top_amounts)*0.01, bar.get_y() + bar.get_height()/2.,
                 f'{amount:.0f}', ha='left', va='center', color='white', fontweight='bold')
    
    # 9. Статистика и сводка
    ax9 = fig.add_subplot(gs[3, :])
    ax9.axis('off')
    
    # Создаем таблицу со статистикой
    stats_data = [
        ['📊 Общая статистика', ''],
        ['💰 Общие расходы:', f'{total:.2f} Тг'],
        ['📈 Средний расход:', f'{avg_expense:.2f} Тг'],
        ['🔥 Максимальный расход:', f'{max_expense:.2f} Тг'],
        ['❄️ Минимальный расход:', f'{min_expense:.2f} Тг'],
        ['🔄 Количество транзакций:', f'{total_transactions}'],
        ['📅 Период:', period_text.capitalize()],
        ['🏷️ Категорий:', f'{len(categories)}']
    ]
    
    table = ax9.table(cellText=stats_data, cellLoc='left', loc='center',
                      colWidths=[0.4, 0.3], cellColours=[['#2d2d2d', '#2d2d2d']] * len(stats_data))
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Стилизация таблицы
    for i in range(len(stats_data)):
        for j in range(2):
            cell = table[(i, j)]
            cell.set_text_props(color='white', weight='bold' if i == 0 else 'normal')
            if i == 0:  # Заголовок
                cell.set_facecolor('#00ff88')
                cell.set_text_props(color='black', weight='bold')
    
    # Общий заголовок дашборда
    fig.suptitle(f'ДАШБОРД РАСХОДОВ ЗА {period_text.upper()}', 
                 fontsize=24, fontweight='bold', color='white', y=0.98)
    
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
    
    # Отправка дашборда и сводки
    await update.message.reply_photo(photo=buf, caption=summary_text, reply_markup=get_main_menu_keyboard())

    # Отправка Excel файла
    await update.message.reply_document(document=excel_buf, filename=f"Отчет_{period_text}.xlsx")
    return ConversationHandler.END

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
    elif text in ["💸 Добавить расход", "📊 Отчеты", "Сегодня", "Неделя", "Месяц", "Год"]:
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
REMINDER_MENU_STATE = 5
REMINDER_TITLE_STATE = 6
REMINDER_DESC_STATE = 7
REMINDER_AMOUNT_STATE = 8
REMINDER_START_DATE_STATE = 9
REMINDER_END_DATE_STATE = 10
REMINDER_MANAGE_STATE = 11

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
        fallbacks=[MessageHandler(filters.TEXT & ~filters.COMMAND, start)],
        allow_reentry=True
    )
    
    # Обработчик для исправления категорий
    correction_conv_handler = ConversationHandler(
        entry_points=[
            MessageHandler(filters.Regex("^🔧 Исправить категории$"), correction_menu),
            CommandHandler("correct", correction_menu)
        ],
        states={
            EXPENSE_CHOICE_STATE: [MessageHandler(filters.Regex("^[0-9]+$"), expense_choice)],
            CATEGORY_CHOICE_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, category_choice)],
            AMOUNT_EDIT_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, amount_edit)],
        },
        fallbacks=[MessageHandler(filters.TEXT & ~filters.COMMAND, start)],
        allow_reentry=True
    )

    # Обработчик для напоминаний
    reminder_conv_handler = ConversationHandler(
        entry_points=[
            MessageHandler(filters.Regex("^⏰ Напоминания$"), reminder_menu),
            CommandHandler("reminders", reminder_menu)
        ],
        states={
            REMINDER_MENU_STATE: [MessageHandler(filters.Regex("^(📝 Добавить напоминание|📋 Список напоминаний)$"), reminder_menu)],
            REMINDER_TITLE_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, reminder_title_input)],
            REMINDER_DESC_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, reminder_desc_input)],
            REMINDER_AMOUNT_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, reminder_amount_input)],
            REMINDER_START_DATE_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, reminder_start_date_input)],
            REMINDER_END_DATE_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, reminder_end_date_input)],
            REMINDER_MANAGE_STATE: [MessageHandler(filters.Regex("^(❌ Удалить \d+|🔙 Назад)$"), reminder_manage)],
        },
        fallbacks=[MessageHandler(filters.TEXT & ~filters.COMMAND, start)],
        allow_reentry=True
    )
    
    application.add_handler(report_conv_handler)
    application.add_handler(correction_conv_handler)
    application.add_handler(reminder_conv_handler)
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

if __name__ == "__main__":
    main() 