import telebot
from telebot import types
import os
import psycopg2
from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt
import io
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import re
from dotenv import load_dotenv

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
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS expenses (
                id SERIAL PRIMARY KEY,
                user_id BIGINT,
                amount REAL,
                currency TEXT,
                description TEXT,
                category TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        ''')
        conn.commit()
        conn.close()
        print("База данных инициализирована (таблица 'expenses' проверена/создана).")

# --- UI (User Interface) ---
def get_main_menu_keyboard():
    keyboard = types.ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    btn_add_expense = types.KeyboardButton('💸 Добавить расход')
    btn_report = types.KeyboardButton('📊 Отчеты')
    keyboard.add(btn_add_expense, btn_report)
    return keyboard

def get_report_period_keyboard():
    keyboard = types.InlineKeyboardMarkup(row_width=2)
    btn1 = types.InlineKeyboardButton("Сегодня", callback_data='report_сегодня')
    btn2 = types.InlineKeyboardButton("Неделя", callback_data='report_неделя')
    btn3 = types.InlineKeyboardButton("Месяц", callback_data='report_месяц')
    btn4 = types.InlineKeyboardButton("Год", callback_data='report_год')
    btn5 = types.InlineKeyboardButton("Другой период...", callback_data='report_другой')
    keyboard.add(btn1, btn2, btn3, btn4, btn5)
    return keyboard

# --- Команды бота ---
@bot.message_handler(commands=['start', 'help', 'menu'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Я твой помощник по учету расходов. Выбери опцию ниже:", reply_markup=get_main_menu_keyboard())

# --- Основные хэндлеры для меню ---
@bot.message_handler(regexp='^💸 Добавить расход$')
def handle_add_expense_menu(message):
    bot.send_message(message.chat.id, "Отлично! Просто напиши 'описание сумма', например: 'хлеб 100, молоко 500'.", reply_markup=types.ReplyKeyboardRemove())

@bot.message_handler(regexp='^📊 Отчеты$')
def handle_report_menu(message):
    bot.send_message(message.chat.id, "За какой период вы хотите отчет?", reply_markup=get_report_period_keyboard())

# --- Callback хэндлеры для кнопок ---
@bot.callback_query_handler(func=lambda call: call.data.startswith('report_'))
def handle_report_callback(call):
    chat_id = call.message.chat.id
    user_id = call.from_user.id
    period_text = call.data.replace('report_', '')
    
    # Сразу удаляем inline-клавиатуру после выбора
    bot.edit_message_reply_markup(chat_id, call.message.message_id, reply_markup=None)
    
    if period_text == 'другой':
        bot.send_message(chat_id, "Введите свой период (например, 'с 01.01.2024 по 31.01.2024').", reply_markup=get_main_menu_keyboard())
        bot.register_next_step_handler(call.message, process_report_period_final)
    else:
        message = call.message
        message.text = period_text
        process_report_period_final(message)

def process_report_period_final(message):
    chat_id = message.chat.id
    user_id = message.from_user.id
    text = message.text.lower()
    
    start_date, end_date = parse_date_period(text)
    # --- ОТЛАДКА: выводим user_id, start_date, end_date, текущее время ---
    print(f"[DEBUG] Отчет: user_id={user_id}, start_date={start_date}, end_date={end_date}, now={datetime.now()}")
    if not start_date:
        bot.send_message(chat_id, "Не могу распознать период.", reply_markup=get_main_menu_keyboard())
        return

    conn = get_db_connection()
    if not conn:
        bot.send_message(chat_id, "Проблема с подключением к базе данных.", reply_markup=get_main_menu_keyboard())
        return
    
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT category, SUM(amount)
            FROM expenses
            WHERE user_id = %s AND timestamp BETWEEN %s AND %s
            GROUP BY category
        ''', (user_id, start_date, end_date))
        data = cursor.fetchall()
    except Exception as e:
        bot.send_message(chat_id, f"Произошла ошибка при получении отчета: {e}", reply_markup=get_main_menu_keyboard())
        return
    finally:
        if conn: conn.close()

    if not data:
        bot.send_message(chat_id, "За выбранный период нет расходов.", reply_markup=get_main_menu_keyboard())
        return

    categories = [row[0] for row in data]
    amounts = [row[1] for row in data]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(amounts, labels=categories, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
    ax.axis('equal')
    report_title_period = text.capitalize()
    if 'с ' in text.lower() and ' по ' in text.lower() and start_date and end_date:
        report_title_period = f"с {start_date.strftime('%d.%m.%Y')} по {end_date.strftime('%d.%m.%Y')}"
    ax.set_title(f'Отчет о расходах за {report_title_period.capitalize()} (Тг)')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    bot.send_photo(chat_id, buf, reply_markup=get_main_menu_keyboard())

def parse_date_period(text):
    text_lower = text.lower()
    start_date = None
    end_date = datetime.now()
    if 'сегодня' in text_lower:
        start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=1, microseconds=-1)
    elif 'неделя' in text_lower:
        start_date = datetime.now() - timedelta(weeks=1)
        start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = datetime.now()
    elif 'месяц' in text_lower:
        start_date = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_date = datetime.now()
    elif 'год' in text_lower:
        start_date = datetime.now().replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        end_date = datetime.now()
    elif 'с ' in text_lower and ' по ' in text_lower:
        try:
            date_from_match = re.search(r'с\s+(\d{1,2}[.]\d{1,2}(?:[.]\d{2,4})?)', text_lower)
            date_to_match = re.search(r'по\s+(\d{1,2}[.]\d{1,2}(?:[.]\d{2,4})?)', text_lower)
            if date_from_match and date_to_match:
                date_from_str = date_from_match.group(1)
                date_to_str = date_to_match.group(1)
                for fmt in ["%d.%m.%Y", "%d.%m.%y", "%d.%m"]:
                    try:
                        start_date = datetime.strptime(date_from_str, fmt)
                        if fmt == "%d.%m": start_date = start_date.replace(year=datetime.now().year)
                        start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
                        break
                    except ValueError: pass
                for fmt in ["%d.%m.%Y", "%d.%m.%y", "%d.%m"]:
                    try:
                        end_date = datetime.strptime(date_to_str, fmt)
                        if fmt == "%d.%m": end_date = end_date.replace(year=datetime.now().year)
                        end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)
                        break
                    except ValueError: pass
        except Exception as e:
            print(f"Ошибка парсинга диапазона дат: {e}")
            start_date = None
    else:
        try:
            month_map = {
                'январь': 1, 'февраль': 2, 'март': 3, 'апрель': 4, 'май': 5, 'июнь': 6,
                'июль': 7, 'август': 8, 'сентябрь': 9, 'октябрь': 10, 'ноябрь': 11, 'декабрь': 12
            }
            month_year_match = re.search(r'([а-яё]+)\s*(\d{4})', text_lower)
            if month_year_match:
                month_name = month_year_match.group(1)
                year = int(month_year_match.group(2))
                month_num = month_map.get(month_name)
                if month_num:
                    start_date = datetime(year, month_num, 1, 0, 0, 0, 0)
                    if month_num == 12: end_date = datetime(year + 1, 1, 1, 0, 0, 0, 0) - timedelta(microseconds=1)
                    else: end_date = datetime(year, month_num + 1, 1, 0, 0, 0, 0) - timedelta(microseconds=1)
        except Exception as e:
            print(f"Ошибка парсинга 'месяц год': {e}")
            pass
    return start_date, end_date


# --- Хэндлер для текстовых сообщений (добавление расходов) ---
@bot.message_handler(func=lambda message: True, content_types=['text'])
def handle_text_messages(message):
    user_id = message.from_user.id
    
    if message.text in ['💸 Добавить расход', '📊 Отчеты']:
        return

    text = message.text
    pattern = r'([\w\s]+)\s+([\d\s.,]+)(?:тг)?'
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    if not matches:
        bot.send_message(message.chat.id, "Не могу распознать расход. Пожалуйста, используйте формат 'описание сумма', например: 'хлеб 100, молоко 500'.", reply_markup=get_main_menu_keyboard())
        return
    
    conn = get_db_connection()
    if not conn:
        bot.send_message(message.chat.id, "Проблема с подключением к базе данных.", reply_markup=get_main_menu_keyboard())
        return

    try:
        success_count = 0
        for match in matches:
            description = match[0].strip()
            amount_str = match[1].strip().replace(' ', '').replace(',', '.')
            
            try:
                amount = float(amount_str)
                currency = 'тг'
                category = classify_expense(description)
                
                # --- ОТЛАДКА: выводим user_id, description, amount, текущее время ---
                print(f"[DEBUG] Добавляю расход: user_id={user_id}, description='{description}', amount={amount}, category='{category}', timestamp={datetime.now()}")
                
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO expenses (user_id, amount, currency, description, category) VALUES (%s, %s, %s, %s, %s)",
                    (user_id, amount, currency, description, category)
                )
                success_count += 1
            except ValueError:
                bot.send_message(message.chat.id, f"Не могу распознать сумму для '{description}'. Проверьте формат.", reply_markup=get_main_menu_keyboard())
                conn.rollback()
                return

        conn.commit()
        if success_count > 0:
            bot.send_message(message.chat.id, f"✅ Успешно добавлено {success_count} расходов.", reply_markup=get_main_menu_keyboard())
    except Exception as e:
        conn.rollback()
        bot.send_message(message.chat.id, f"Произошла ошибка при сохранении расхода: {e}", reply_markup=get_main_menu_keyboard())
    finally:
        conn.close()


# --- Запуск бота ---
if __name__ == '__main__':
    train_model(TRAINING_DATA)
    init_db()
    
    commands = [
        telebot.types.BotCommand("/start", "Перезапустить бота"),
        telebot.types.BotCommand("/menu", "Показать главное меню"),
        telebot.types.BotCommand("/report", "Получить отчет о расходах"),
    ]
    bot.set_my_commands(commands)
    
    print("Бот запущен и готов принимать сообщения...")
    bot.polling(none_stop=True)