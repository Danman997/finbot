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
        "продукты","продукт","продуктовый","прод","еда","питание","бакалея","молочка","выпечка","овощи","фрукты"
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
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS expenses (
                id SERIAL PRIMARY KEY,
                amount NUMERIC(10, 2) NOT NULL,
                description TEXT,
                category VARCHAR(100) NOT NULL,
                transaction_date TIMESTAMP WITH TIME ZONE NOT NULL
            );
        ''')
        conn.commit()
        conn.close()
        logger.info("База данных инициализирована (таблица 'expenses' проверена/создана).")

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

# --- UI (User Interface) ---
def get_main_menu_keyboard():
    keyboard = [
        [KeyboardButton("💸 Добавить расход"), KeyboardButton("📊 Отчеты")],
        [KeyboardButton("🔧 Исправить категории"), KeyboardButton("📚 Обучить модель")]
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
        grouped_data = df.groupby('Категория', as_index=False)['Сумма'].sum().sort_values(by='Сумма', ascending=False)
        categories = grouped_data['Категория'].tolist()
        amounts = grouped_data['Сумма'].tolist()
        total = df['Сумма'].sum()
    except Exception as e:
        logger.error(f"Ошибка при создании DataFrame: {e}")
        return ConversationHandler.END

    # Создание Excel файла
    excel_buf = io.BytesIO()
    df.to_excel(excel_buf, index=False, engine='xlsxwriter')
    excel_buf.seek(0)

    # График
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(amounts, labels=None, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
    ax.axis('equal')
    plt.title(f'Отчет о расходах за {period_text.capitalize()} (Тг)')

    # Легенда снизу
    legend_labels = [f"{cat} — {amt:.2f} Тг" for cat, amt in zip(categories, amounts)]
    plt.legend(wedges, legend_labels, title="Категории", loc="lower center", bbox_to_anchor=(0.5, -0.15), fontsize=12)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close(fig)

    # Текстовая таблица для подписи
    table_text = "\n".join([f"{cat}: {amt:.2f} Тг" for cat, amt in zip(categories, amounts)])
    table_text += f"\n\nИтого: {total:.2f} Тг"

    # Отправка графика и подписи
    await update.message.reply_photo(photo=buf, caption=table_text, reply_markup=get_main_menu_keyboard())

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
    
    application.add_handler(report_conv_handler)
    application.add_handler(correction_conv_handler)
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

    # Планирование ежедневного обучения
    schedule.every().day.at("00:00").do(daily_training)

    # Запуск планировщика в отдельном потоке
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main() 