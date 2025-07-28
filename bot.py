import telebot
from telebot import types
import os
import logging
import psycopg2
from psycopg2 import sql
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, ConversationHandler, filters
import matplotlib.pyplot as plt
import io
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import re

# Загружаем переменные окружения из .env файла
load_dotenv()

# --- Настройки бота ---
BOT_TOKEN = os.environ.get('BOT_TOKEN') 
if not BOT_TOKEN:
    print("Ошибка: Токен бота не найден. Установите переменную окружения BOT_TOKEN.")
    exit()
bot = telebot.TeleBot(BOT_TOKEN)

DATABASE_URL = os.environ.get('DATABASE_URL')
if not DATABASE_URL:
    print("Ошибка: URL базы данных не найден. Установите переменную окружения DATABASE_URL.")
    exit()

# --- Модель классификации (scikit-learn) ---
vectorizer = TfidfVectorizer()
classifier = LogisticRegression(max_iter=1000)

TRAINING_DATA = [
    ("хлеб", "Еда"), ("молоко", "Еда"), ("яйца", "Еда"), ("фрукты", "Еда"),
    ("овощи", "Еда"), ("продукты", "Еда"), ("обед", "Еда"), ("ужин", "Еда"),
    ("кофе", "Еда"), ("сок", "Еда"), ("чай", "Еда"), ("вода", "Еда"),
    ("булочка", "Еда"), ("пицца", "Еда"), ("рыба", "Еда"), ("мясо", "Еда"),
    ("колбаса", "Еда"), ("сыр", "Еда"), ("рис", "Еда"), ("картошка", "Еда"),
    ("сладости", "Еда"), ("конфеты", "Еда"), ("шоколад", "Еда"), ("кефир", "Еда"),
    ("сметана", "Еда"), ("йогурт", "Еда"), ("салат", "Еда"), ("мороженое", "Еда"),
    ("завтрак", "Еда"), ("ланч", "Еда"),
    
    ("бензин", "Транспорт"), ("такси", "Транспорт"), ("автобус", "Транспорт"),
    ("метро", "Транспорт"), ("проезд", "Транспорт"), ("поезд", "Транспорт"),
    ("самолет", "Транспорт"), ("маршрутка", "Транспорт"), ("проездной", "Транспорт"),
    ("авто", "Транспорт"), ("ремонт авто", "Транспорт"), ("парковка", "Транспорт"),
    ("штраф", "Транспорт"),
    
    ("билеты", "Развлечения"), ("кино", "Развлечения"), ("театр", "Развлечения"),
    ("концерт", "Развлечения"), ("книга", "Развлечения"), ("игры", "Развлечения"),
    ("аттракционы", "Развлечения"), ("музей", "Развлечения"), ("подписка", "Развлечения"),
    ("бар", "Развлечения"), ("ресторан", "Развлечения"), ("кафе", "Развлечения"),
    ("вечеринка", "Развлечения"), ("поход", "Развлечения"), ("отпуск", "Развлечения"),

    ("одежда", "Одежда"), ("обувь", "Одежда"), ("футболка", "Одежда"),
    ("брюки", "Одежда"), ("платье", "Одежда"), ("куртка", "Одежда"),
    ("кроссовки", "Одежда"), ("свитер", "Одежда"), ("джинсы", "Одежда"),
    ("пальто", "Одежда"), ("шапка", "Одежда"), ("перчатки", "Одежда"),

    ("коммуналка", "Жилье"), ("аренда", "Жилье"), ("свет", "Жилье"),
    ("вода", "Жилье"), ("газ", "Жилье"), ("квитанция", "Жилье"), ("ипотека", "Жилье"),
    ("интернет", "Жилье"), ("телефон", "Жилье"),
    
    ("аптека", "Здоровье"), ("врач", "Здоровье"), ("лекарства", "Здоровье"),
    ("стоматолог", "Здоровье"), ("витамины", "Здоровье"), ("больница", "Здоровье"),
    ("страховка", "Здоровье"), ("клиника", "Здоровье"), ("фитнес", "Здоровье"),
    ("спортзал", "Здоровье"),

    ("связь", "Связь"), ("мобильный", "Связь"), ("тариф", "Связь"),

    ("скотч", "Дом/Канцелярия"), ("ручки", "Дом/Канцелярия"), ("бумага", "Дом/Канцелярия"),
    ("канцелярия", "Дом/Канцелярия"), ("дом", "Дом/Канцелярия"), ("посуда", "Дом/Канцелярия"),
    ("чистящие", "Дом/Канцелярия"), ("инструменты", "Дом/Канцелярия"),
    ("техника", "Дом/Канцелярия"), ("мебель", "Дом/Канцелярия"), ("постельное", "Дом/Канцелярия"),
    ("подарки", "Прочее"), ("другое", "Прочее"), ("разное", "Прочее"), 
    ("налоги", "Прочее"), ("сюрприз", "Прочее")
]

def train_model(data):
    if not data:
        print("Нет данных для обучения модели. Модель не будет обучена.")
        return
    descriptions = [item[0] for item in data]
    categories = [item[1] for item in data]
    X = vectorizer.fit_transform(descriptions)
    classifier.fit(X, categories)
    print("Модель классификации успешно обучена.")

def classify_expense(description):
    try:
        if not hasattr(classifier, 'classes_') or len(classifier.classes_) == 0:
            return 'Прочее'
        description_vectorized = vectorizer.transform([description.lower()])
        prediction = classifier.predict(description_vectorized)[0]
        return prediction
    except Exception as e:
        print(f"Ошибка при классификации: {e}. Возвращаю 'Прочее'.")
        return 'Прочее'

# --- Функции для работы с базой данных ---
def get_db_connection():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        print(f"Ошибка подключения к БД: {e}")
        return None

def init_db():
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        # Проверяем, существует ли столбец family_id, и удаляем его, если он есть
        try:
            cursor.execute("ALTER TABLE expenses DROP COLUMN family_id;")
            conn.commit()
            print("Столбец family_id успешно удален из таблицы expenses.")
        except psycopg2.errors.UndefinedColumn:
            conn.rollback() # Откатываем транзакцию, если столбца не было
            print("Столбец family_id не существует, ничего не делаем.")
        except Exception as e:
            conn.rollback()
            print(f"Произошла ошибка при удалении столбца family_id: {e}")

        # Проверяем, существует ли столбец user_id, и удаляем его, если он есть
        try:
            cursor.execute("ALTER TABLE expenses DROP COLUMN user_id;")
            conn.commit()
            print("Столбец user_id успешно удален из таблицы expenses.")
        except psycopg2.errors.UndefinedColumn:
            conn.rollback() # Откатываем транзакцию, если столбца не было
            print("Столбец user_id не существует, ничего не делаем.")
        except Exception as e:
            conn.rollback()
            print(f"Произошла ошибка при удалении столбца user_id: {e}")

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
        print("База данных инициализирована (таблица 'expenses' проверена/создана).")

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
async def start(update, context) -> None:
    """Отвечает на команду /start."""
    await update.message.reply_text(
        "Привет! Я твой помощник по учету расходов. Выбери опцию ниже:",
        reply_markup=get_main_menu_keyboard()
    )

# --- Основные хэндлеры для меню ---
async def handle_add_expense_menu(update, context) -> None:
    await update.message.reply_text(
        "Отлично! Просто напиши 'описание сумма', например: 'хлеб 100, молоко 500'.",
        reply_markup=ReplyKeyboardRemove()
    )

async def handle_report_menu(update, context) -> None:
    await update.message.reply_text(
        "За какой период вы хотите отчет?",
        reply_markup=get_report_period_keyboard()
    )

# --- Callback хэндлеры для кнопок ---
async def handle_report_callback(update, context) -> None:
    # Этот хэндлер будет обрабатывать нажатия кнопок отчета из get_report_period_keyboard
    # В ConversationHandler entry_points его не было, поэтому он должен быть добавлен.
    period_text = update.message.text.lower()
    
    start_date, end_date = parse_date_period(period_text)
    if not start_date:
        await update.message.reply_text("Не могу распознать период.", reply_markup=get_main_menu_keyboard())
        return

    conn = get_db_connection()
    if not conn:
        await update.message.reply_text("Проблема с подключением к базе данных.", reply_markup=get_main_menu_keyboard())
        return
    
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT category, SUM(amount)
            FROM expenses
            WHERE transaction_date BETWEEN %s AND %s
            GROUP BY category
        ''', (start_date, end_date))
        data = cursor.fetchall()
    except Exception as e:
        await update.message.reply_text(f"Произошла ошибка при получении отчета: {e}", reply_markup=get_main_menu_keyboard())
        return
    finally:
        if conn: conn.close()

    if not data:
        await update.message.reply_text("За выбранный период нет расходов.", reply_markup=get_main_menu_keyboard())
        return

    categories = [row[0] for row in data]
    amounts = [row[1] for row in data]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(sizes=amounts, labels=categories, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
    ax.axis('equal')
    report_title_period = period_text.capitalize()
    if 'с ' in period_text and ' по ' in period_text and start_date and end_date:
        report_title_period = f"с {start_date.strftime('%d.%m.%Y')} по {end_date.strftime('%d.%m.%Y')}"
    plt.title(f'Отчет о расходах за {report_title_period.capitalize()} (Тг)')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close(fig)
    await update.message.reply_photo(photo=buf, caption=f"Отчет за {report_title_period.capitalize()}", reply_markup=get_main_menu_keyboard())


def parse_date_period(text):
    text_lower = text.lower()
    start_date = None
    end_date = datetime.now(timezone.utc) # Использование timezone.utc
    current_time_aware = datetime.now(timezone.utc)

    if 'сегодня' in text_lower:
        start_date = current_time_aware.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=1, microseconds=-1) # До конца дня
    elif 'неделя' in text_lower:
        start_date = (current_time_aware - timedelta(days=current_time_aware.weekday())).replace(hour=0, minute=0, second=0, microsecond=0) # Начало текущей недели (понедельник)
        end_date = current_time_aware.replace(hour=23, minute=59, second=59, microsecond=999999)
    elif 'месяц' in text_lower:
        start_date = current_time_aware.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_date = current_time_aware.replace(hour=23, minute=59, second=59, microsecond=999999)
    elif 'год' in text_lower:
        start_date = current_time_aware.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        end_date = current_time_aware.replace(hour=23, minute=59, second=59, microsecond=999999)
    elif 'с ' in text_lower and ' по ' in text_lower:
        try:
            date_from_match = re.search(r'с\s+(\d{1,2}[.]\d{1,2}(?:[.]\d{2,4})?)', text_lower)
            date_to_match = re.search(r'по\s+(\d{1,2}[.]\d{1,2}(?:[.]\d{2,4})?)', text_lower)
            if date_from_match and date_to_match:
                date_from_str = date_from_match.group(1)
                date_to_str = date_to_match.group(1)
                for fmt in ["%d.%m.%Y", "%d.%m.%y", "%d.%m"]:
                    try:
                        start_date = datetime.strptime(date_from_str, fmt).replace(tzinfo=timezone.utc) # Указываем UTC
                        if fmt == "%d.%m": start_date = start_date.replace(year=current_time_aware.year)
                        start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
                        break
                    except ValueError: pass
                for fmt in ["%d.%m.%Y", "%d.%m.%y", "%d.%m"]:
                    try:
                        end_date = datetime.strptime(date_to_str, fmt).replace(tzinfo=timezone.utc) # Указываем UTC
                        if fmt == "%d.%m": end_date = end_date.replace(year=current_time_aware.year)
                        end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)
                        break
                    except ValueError: pass
        except Exception as e:
            logger.error(f"Ошибка парсинга диапазона дат: {e}")
            start_date = None
    return start_date, end_date


# --- Хэндлер для текстовых сообщений (добавление расходов) ---
async def handle_message(update: Update, context) -> None:
    text = update.message.text.strip()
    
    # Игнорируем сообщения, которые являются кнопками
    if text in ["💸 Добавить расход", "📊 Отчеты", "Сегодня", "Неделя", "Месяц", "Год"]:
        return

    logger.info(f"Получено сообщение от UserID={update.message.from_user.id}: {text}")

    # Пробуем распарсить как "Описание Сумма"
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
        
        category = classify_expense(description) # Используем вашу функцию classify_expense
        
        transaction_date = datetime.now(timezone.utc) # Сохраняем в UTC
        
        # user_id больше не передается, так как его нет в таблице expenses
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

def main():
    """Запускает бота."""
    if not BOT_TOKEN:
        logger.error("BOT_TOKEN environment variable is not set. Bot cannot start.")
        raise ValueError("BOT_TOKEN is not set.")

    # Обучаем модель перед запуском бота
    train_model(TRAINING_DATA)
    # Создаем/обновляем таблицу в БД
    init_db() 

    application = Application.builder().token(BOT_TOKEN).build()

    # ConversationHandler для меню отчетов
    report_conv_handler = ConversationHandler(
        entry_points=[
            MessageHandler(filters.Regex("^📊 Отчеты$"), report_menu), # Нажатие на кнопку "Отчеты"
            CommandHandler("report", report_menu) # Если пользователь введет /report вручную
        ],
        states={
            PERIOD_CHOICE_STATE: [MessageHandler(filters.Regex("^(Сегодня|Неделя|Месяц|Год)$"), period_choice)],
        },
        fallbacks=[MessageHandler(filters.TEXT & ~filters.COMMAND, lambda u, c: c.bot.send_message(u.effective_chat.id, "Выбор периода отменен. Выберите опцию:", reply_markup=get_main_menu_keyboard()))],
        allow_reentry=True
    )
    application.add_handler(report_conv_handler)
    
    # Обработчики команд и сообщений
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)) # Обработчик для всех текстовых сообщений

    logger.info("Бот запущен!")
    application.run_polling()

if __name__ == "__main__":
    main()