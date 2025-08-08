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
logging.basicConfig(level=logging.INFO)
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

# --- Модель классификации (scikit-learn) ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

vectorizer = TfidfVectorizer()
classifier = LogisticRegression(max_iter=1000)

TRAINING_DATA = [
    ("хлеб", "Продукты"), ("батон", "Продукты"), ("булочка", "Продукты"), ("багет", "Продукты"), ("лаваш", "Продукты"),
    ("пицца", "Продукты"), ("пирог", "Продукты"), ("пирожок", "Продукты"), ("печенье", "Продукты"), ("торт", "Продукты"),
    ("круассан", "Продукты"), ("бублик", "Продукты"), ("сухарики", "Продукты"), ("пряники", "Продукты"), ("крекер", "Продукты"),
    ("молоко", "Продукты"), ("кефир", "Продукты"), ("сливки", "Продукты"), ("сметана", "Продукты"), ("йогурт", "Продукты"),
    ("творог", "Продукты"), ("сыр", "Продукты"), ("масло сливочное", "Продукты"), ("масло подсолнечное", "Продукты"), ("маргарин", "Продукты"),
    ("яйца", "Продукты"), ("мясо", "Продукты"), ("говядина", "Продукты"), ("свинина", "Продукты"), ("баранина", "Продукты"),
    ("курица", "Продукты"), ("индейка", "Продукты"), ("утка", "Продукты"), ("рыба", "Продукты"), ("лосось", "Продукты"),
    ("форель", "Продукты"), ("треска", "Продукты"), ("минтай", "Продукты"), ("тунец", "Продукты"), ("икра", "Продукты"),
    ("колбаса", "Продукты"), ("сосиски", "Продукты"), ("сардельки", "Продукты"), ("бекон", "Продукты"), ("шашлык", "Продукты"),
    ("консервы", "Продукты"), ("тушенка", "Продукты"), ("паштет", "Продукты"), ("гречка", "Продукты"), ("рис", "Продукты"),
    ("перловка", "Продукты"), ("овсянка", "Продукты"), ("пшено", "Продукты"), ("манка", "Продукты"), ("макароны", "Продукты"),
    ("вермишель", "Продукты"), ("спагетти", "Продукты"), ("лапша", "Продукты"), ("чипсы", "Продукты"), ("орехи", "Продукты"),
    ("арахис", "Продукты"), ("миндаль", "Продукты"), ("фисташки", "Продукты"), ("грецкий орех", "Продукты"), ("яблоки", "Продукты"),
    ("бананы", "Продукты"), ("апельсины", "Продукты"), ("мандарины", "Продукты"), ("груши", "Продукты"), ("виноград", "Продукты"),
    ("персики", "Продукты"), ("абрикосы", "Продукты"), ("сливы", "Продукты"), ("киви", "Продукты"), ("лимоны", "Продукты"),
    ("картофель", "Продукты"), ("морковь", "Продукты"), ("свекла", "Продукты"), ("лук", "Продукты"), ("чеснок", "Продукты"),
    ("капуста", "Продукты"), ("огурцы", "Продукты"), ("помидоры", "Продукты"), ("перец", "Продукты"), ("баклажаны", "Продукты"),
    ("кабачки", "Продукты"), ("тыква", "Продукты"), ("укроп", "Продукты"), ("петрушка", "Продукты"), ("салат", "Продукты"),
    ("шпинат", "Продукты"), ("зелень", "Продукты"), ("сахар", "Продукты"), ("соль", "Продукты"), ("перец молотый", "Продукты"),
    ("приправы", "Продукты"), ("кетчуп", "Продукты"), ("майонез", "Продукты"), ("горчица", "Продукты"), ("продукты", "Продукты"),
    ("продуктовый", "Продукты"), ("продукт", "Продукты"), ("прод", "Продукты"), ("еда", "Продукты"), ("питание", "Продукты"),
    ("футболка", "Одежда"), ("рубашка", "Одежда"), ("кофта", "Одежда"), ("свитер", "Одежда"), ("толстовка", "Одежда"),
    ("пиджак", "Одежда"), ("жилет", "Одежда"), ("пальто", "Одежда"), ("куртка", "Одежда"), ("плащ", "Одежда"),
    ("шуба", "Одежда"), ("брюки", "Одежда"), ("джинсы", "Одежда"), ("шорты", "Одежда"), ("юбка", "Одежда"),
    ("платье", "Одежда"), ("комбинезон", "Одежда"), ("колготки", "Одежда"), ("носки", "Одежда"), ("гетры", "Одежда"),
    ("обувь", "Одежда"), ("ботинки", "Одежда"), ("туфли", "Одежда"), ("кроссовки", "Одежда"), ("кеды", "Одежда"),
    ("сланцы", "Одежда"), ("тапочки", "Одежда"), ("сандалии", "Одежда"), ("одежда", "Одежда"), ("одежке", "Одежда"),
    ("одёжка", "Одежда"), ("шмот", "Одежда"), ("шмотки", "Одежда"), ("вещи", "Одежда"),
    # ...additional categories and items...
]

def train_model(data):
    if not data:
        logger.warning("Нет данных для обучения модели. Модель не будет обучена.")
        return
    descriptions = [item[0] for item in data]
    categories = [item[1] for item in data]
    X = vectorizer.fit_transform(descriptions)
    classifier.fit(X, categories)
    logger.info("Модель классификации успешно обучена.")

# Обучение модели с обновленными данными
train_model(TRAINING_DATA)

def classify_expense(description):
    try:
        if not hasattr(classifier, 'classes_') or len(classifier.classes_) == 0:
            return 'Прочее'
        description_vectorized = vectorizer.transform([description.lower()])
        prediction = classifier.predict(description_vectorized)[0]
        return prediction
    except Exception as e:
        logger.error(f"Ошибка при классификации: {e}. Возвращаю 'Прочее'.")
        return 'Прочее'

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

# --- UI (User Interface) ---
def get_main_menu_keyboard():
    keyboard = [
        [KeyboardButton("💸 Добавить расход"), KeyboardButton("📊 Отчеты")]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

def get_report_period_keyboard():
    keyboard = [
        [KeyboardButton("Сегодня"), KeyboardButton("Неделя")],
        [KeyboardButton("Месяц"), KeyboardButton("Год")]
    ]
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
    except Exception as e:
        logger.error(f"Ошибка при создании DataFrame: {e}")
        return ConversationHandler.END

    # Создание Excel файла
    excel_buf = io.BytesIO()
    df.to_excel(excel_buf, index=False, engine='xlsxwriter')
    excel_buf.seek(0)

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
    if text in ["💸 Добавить расход", "📊 Отчеты", "Сегодня", "Неделя", "Месяц", "Год"]:
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
                f"✅ Расход '{description}' ({amount:.2f}) записан в категорию '{category}'!",
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

def main():
    train_model(TRAINING_DATA)
    init_db()
    application = Application.builder().token(BOT_TOKEN).build()

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
    application.add_handler(report_conv_handler)
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("Бот запущен!")
    application.run_polling()

    # Функция для ежедневного обучения модели
    def daily_training():
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute('SELECT description, category FROM expenses')
            data = cursor.fetchall()
            conn.close()
            if data:
                train_model(data)
                logger.info("Модель успешно обучена с новыми данными.")
            else:
                logger.warning("Нет новых данных для обучения модели.")
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