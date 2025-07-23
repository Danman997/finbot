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
import random
import string
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

ADMIN_USER_ID = 498410375

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
            CREATE TABLE IF NOT EXISTS families (
                id SERIAL PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                owner_user_id BIGINT NULL,
                subscription_end_date TIMESTAMP NULL,
                is_active BOOLEAN DEFAULT FALSE,
                invite_code TEXT UNIQUE NULL
            );
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_families (
                user_id BIGINT NOT NULL,
                family_id INTEGER NOT NULL REFERENCES families(id) ON DELETE CASCADE,
                role TEXT DEFAULT 'member',
                PRIMARY KEY (user_id, family_id)
            );
        ''')
        conn.commit()
        conn.close()
        print("База данных инициализирована (таблицы проверены/созданы).")

def get_expense_table_name(family_id):
    return f"expenses_family_{family_id}"

def create_expense_table(family_id, conn):
    cursor = conn.cursor()
    table_name = get_expense_table_name(family_id)
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {table_name} (
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

# --- Вспомогательные функции ---
def get_user_active_family_id(user_id):
    conn = get_db_connection()
    if not conn: return None
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT family_id FROM user_families WHERE user_id = %s ORDER BY family_id LIMIT 1
        """, (user_id,))
        family_id = cursor.fetchone()
        return family_id[0] if family_id else None
    except Exception as e:
        print(f"Ошибка при получении family_id: {e}")
        return None
    finally:
        if conn: conn.close()

def is_user_registered(user_id):
    conn = get_db_connection()
    if not conn: return False
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM user_families WHERE user_id = %s", (user_id,))
    is_registered = cursor.fetchone()
    conn.close()
    return is_registered is not None

def is_family_subscription_active(family_id):
    if not family_id:
        return False
    conn = get_db_connection()
    if not conn: return False
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT subscription_end_date, is_active FROM families WHERE id = %s",
            (family_id,)
        )
        result = cursor.fetchone()
        if result:
            end_date, active_flag = result
            return active_flag and end_date and end_date > datetime.now()
        return False
    except Exception as e:
        print(f"Ошибка при проверке подписки: {e}")
        return False
    finally:
        if conn: conn.close()

# --- UI (User Interface) ---
def get_main_menu_keyboard():
    keyboard = types.ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    btn_add_expense = types.KeyboardButton('💸 Добавить расход')
    btn_report = types.KeyboardButton('📊 Отчеты')
    btn_manage_family = types.KeyboardButton('👨‍👩‍👧‍👦 Семья')
    keyboard.add(btn_add_expense, btn_report, btn_manage_family)
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
    user_id = message.from_user.id
    if not is_user_registered(user_id):
        bot.reply_to(message, "Привет! Вы пока не зарегистрированы. Пожалуйста, обратитесь к администратору, чтобы он добавил вас.", reply_markup=types.ReplyKeyboardRemove())
    else:
        bot.reply_to(message, "С возвращением! Выбери опцию ниже:", reply_markup=get_main_menu_keyboard())

# --- Основные хэндлеры для меню ---
@bot.message_handler(regexp='^💸 Добавить расход$')
def handle_add_expense_menu(message):
    user_id = message.from_user.id
    family_id = get_user_active_family_id(user_id)
    if family_id is None:
        bot.send_message(message.chat.id, "Вы не состоите в активной семье. Для записи расходов вам нужно быть добавленным администратором. Напишите /start.", reply_markup=get_main_menu_keyboard())
        return
    bot.send_message(message.chat.id, "Отлично! Просто напиши 'описание сумма', например: 'хлеб 100, молоко 500'.", reply_markup=types.ReplyKeyboardRemove())

@bot.message_handler(regexp='^📊 Отчеты$')
def handle_report_menu(message):
    user_id = message.from_user.id
    family_id = get_user_active_family_id(user_id)
    if family_id is None:
        bot.send_message(message.chat.id, "Вы не состоите в активной семье. Для получения отчетов нужна семья.", reply_markup=get_main_menu_keyboard())
        return
    bot.send_message(message.chat.id, "За какой период вы хотите отчет?", reply_markup=get_report_period_keyboard())

@bot.message_handler(regexp='^👨‍👩‍👧‍👦 Семья$')
def handle_family_menu(message):
    user_id = message.from_user.id
    family_id = get_user_active_family_id(user_id)
    if family_id is None:
        bot.send_message(message.chat.id, "Вы пока не состоите ни в одной семье.", reply_markup=get_main_menu_keyboard())
        return
    
    keyboard = types.InlineKeyboardMarkup(row_width=1)
    btn1 = types.InlineKeyboardButton("📝 Информация о семье", callback_data='my_family_info')
    btn2 = types.InlineKeyboardButton("🚶 Выйти из семьи", callback_data='leave_family_confirm')
    btn3 = types.InlineKeyboardButton("👥 Члены семьи", callback_data='family_members')
    btn4 = types.InlineKeyboardButton("🔗 Пригласить в семью", callback_data='invite_member')
    btn5 = types.InlineKeyboardButton("✏️ Изменить название семьи", callback_data='edit_family_name')
    keyboard.add(btn1, btn2, btn3, btn4, btn5)
    bot.send_message(message.chat.id, "Что вы хотите сделать с семьей?", reply_markup=keyboard)

# --- Callback хэндлеры для кнопок ---
@bot.callback_query_handler(func=lambda call: call.data.startswith('report_'))
def handle_report_callback(call):
    chat_id = call.message.chat.id
    user_id = call.from_user.id
    period_text = call.data.replace('report_', '')
    
    bot.edit_message_reply_markup(chat_id, call.message.message_id)
    
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
    
    family_id = get_user_active_family_id(user_id)
    if family_id is None:
        bot.send_message(chat_id, "Произошла ошибка: не удалось определить активную семью для отчета.", reply_markup=get_main_menu_keyboard())
        return

    start_date, end_date = parse_date_period(text)
    if not start_date:
        bot.send_message(chat_id, "Не могу распознать период.", reply_markup=get_main_menu_keyboard())
        return

    conn = get_db_connection()
    if not conn:
        bot.send_message(chat_id, "Проблема с подключением к базе данных.", reply_markup=get_main_menu_keyboard())
        return
    
    table_name = get_expense_table_name(family_id)
    try:
        cursor = conn.cursor()
        cursor.execute(f'''
            SELECT category, SUM(amount)
            FROM {table_name}
            WHERE timestamp BETWEEN %s AND %s
            GROUP BY category
        ''', (start_date, end_date))
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

@bot.callback_query_handler(func=lambda call: call.data == 'leave_family_confirm')
def handle_leave_family_confirm(call):
    keyboard = types.InlineKeyboardMarkup(row_width=2)
    btn_yes = types.InlineKeyboardButton("Да, я уверен", callback_data='leave_family_yes')
    btn_no = types.InlineKeyboardButton("Нет, отмена", callback_data='leave_family_no')
    keyboard.add(btn_yes, btn_no)
    bot.edit_message_text("Вы уверены, что хотите выйти из семьи? Все ваши расходы останутся, но вы потеряете доступ к общим отчетам.", call.message.chat.id, call.message.message_id, reply_markup=keyboard)

@bot.callback_query_handler(func=lambda call: call.data == 'leave_family_yes')
def handle_leave_family_yes(call):
    user_id = call.from_user.id
    family_id = get_user_active_family_id(user_id)
    if not family_id:
        bot.edit_message_text("Вы не состоите ни в одной семье.", call.message.chat.id, call.message.message_id, reply_markup=get_main_menu_keyboard())
        return

    conn = get_db_connection()
    if not conn:
        bot.edit_message_text("Проблема с подключением к базе данных.", call.message.chat.id, call.message.message_id, reply_markup=get_main_menu_keyboard())
        return
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user_families WHERE user_id = %s AND family_id = %s", (user_id, family_id))
        conn.commit()
        bot.edit_message_text("Вы успешно вышли из семьи. Чтобы снова добавлять расходы, вам нужно написать /start.", call.message.chat.id, call.message.message_id, reply_markup=get_main_menu_keyboard())
    except Exception as e:
        conn.rollback()
        bot.edit_message_text(f"Произошла ошибка при выходе из семьи: {e}", call.message.chat.id, call.message.message_id)
    finally:
        conn.close()

@bot.callback_query_handler(func=lambda call: call.data == 'leave_family_no')
def handle_leave_family_no(call):
    bot.edit_message_text("Выход из семьи отменен.", call.message.chat.id, call.message.message_id, reply_markup=get_main_menu_keyboard())

@bot.callback_query_handler(func=lambda call: call.data == 'family_members')
def handle_family_members(call):
    user_id = call.from_user.id
    family_id = get_user_active_family_id(user_id)
    if not family_id:
        bot.edit_message_text("Вы не состоите ни в одной семье.", call.message.chat.id, call.message.message_id)
        return
    
    conn = get_db_connection()
    if not conn:
        bot.edit_message_text("Проблема с подключением к базе данных.", call.message.chat.id, call.message.message_id)
        return
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT user_id, role
            FROM user_families
            WHERE family_id = %s
        """, (family_id,))
        members = cursor.fetchall()
        
        response = "👥 **Члены вашей семьи:**\n\n"
        for member in members:
            member_id, role = member
            try:
                member_chat = bot.get_chat(member_id)
                name = member_chat.first_name if member_chat.first_name else f"Пользователь {member_id}"
                if member_chat.username:
                    name += f" (@{member_chat.username})"
            except Exception:
                name = f"Пользователь {member_id}"
            
            response += f"🔹 {name} ({role.capitalize()})\n"
        
        bot.edit_message_text(response, call.message.chat.id, call.message.message_id, parse_mode='Markdown', reply_markup=get_main_menu_keyboard())
    except Exception as e:
        bot.edit_message_text(f"Произошла ошибка: {e}", call.message.chat.id, call.message.message_id)
    finally:
        conn.close()

@bot.callback_query_handler(func=lambda call: call.data == 'edit_family_name')
def handle_edit_family_name(call):
    user_id = call.from_user.id
    family_id = get_user_active_family_id(user_id)
    
    conn = get_db_connection()
    if not conn:
        bot.edit_message_text("Проблема с подключением к базе данных.", call.message.chat.id, call.message.message_id)
        return
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT role FROM user_families WHERE user_id = %s AND family_id = %s", (user_id, family_id))
        user_role = cursor.fetchone()
        
        if user_role and user_role[0] == 'admin':
            bot.edit_message_text("Введите новое название для вашей семьи:", call.message.chat.id, call.message.message_id)
            bot.register_next_step_handler(call.message, edit_family_name_final)
        else:
            bot.edit_message_text("Только администратор семьи может изменить ее название.", call.message.chat.id, call.message.message_id, reply_markup=get_main_menu_keyboard())
    except Exception as e:
        bot.edit_message_text(f"Произошла ошибка: {e}", call.message.chat.id, call.message.message_id)
    finally:
        conn.close()

def edit_family_name_final(message):
    new_name = message.text.strip()
    user_id = message.from_user.id
    family_id = get_user_active_family_id(user_id)
    
    conn = get_db_connection()
    if not conn:
        bot.send_message(message.chat.id, "Проблема с подключением к базе данных.", reply_markup=get_main_menu_keyboard())
        return
    try:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE families SET name = %s WHERE id = %s",
            (new_name, family_id)
        )
        conn.commit()
        bot.send_message(message.chat.id, f"Название семьи успешно изменено на '{new_name}'.", reply_markup=get_main_menu_keyboard())
    except Exception as e:
        conn.rollback()
        bot.send_message(message.chat.id, f"Произошла ошибка при изменении названия: {e}", reply_markup=get_main_menu_keyboard())
    finally:
        conn.close()

@bot.callback_query_handler(func=lambda call: call.data == 'invite_member')
def handle_invite_member(call):
    user_id = call.from_user.id
    family_id = get_user_active_family_id(user_id)
    
    conn = get_db_connection()
    if not conn:
        bot.edit_message_text("Проблема с подключением к базе данных.", call.message.chat.id, call.message.message_id)
        return

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT invite_code FROM families WHERE id = %s", (family_id,))
        invite_code = cursor.fetchone()[0]
        
        bot.edit_message_text(
            f"🔗 Чтобы пригласить человека, отправьте ему этот код: `{invite_code}`\n\n"
            f"Для присоединения ему нужно будет просто отправить этот код боту.",
            call.message.chat.id, call.message.message_id, parse_mode='Markdown'
        )
    except Exception as e:
        bot.edit_message_text(f"Произошла ошибка при получении кода: {e}", call.message.chat.id, call.message.message_id)
    finally:
        conn.close()

@bot.callback_query_handler(func=lambda call: call.data == 'my_family_info')
def handle_my_family_info(call):
    user_id = call.from_user.id
    conn = get_db_connection()
    if not conn:
        bot.send_message(call.message.chat.id, "Проблема с подключением к базе данных.", reply_markup=get_main_menu_keyboard())
        return
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT f.id, f.name, f.invite_code
            FROM families f
            JOIN user_families uf ON f.id = uf.family_id
            WHERE uf.user_id = %s
        """, (user_id,))
        family_info = cursor.fetchone()
    except Exception as e:
        bot.send_message(call.message.chat.id, f"Произошла ошибка: {e}", reply_markup=get_main_menu_keyboard())
        return
    finally:
        if conn: conn.close()
    
    if family_info:
        family_id, family_name, invite_code = family_info
        
        message_text = (
            f"**Моя семья:**\n"
            f"Имя: {family_name}\n"
            f"ID: {family_id}\n"
            f"Код приглашения: `{invite_code}`\n"
        )
        bot.edit_message_text(message_text, call.message.chat.id, call.message.message_id, parse_mode='Markdown', reply_markup=get_main_menu_keyboard())
    else:
        bot.edit_message_text("Вы пока не состоите ни в одной семье.", call.message.chat.id, call.message.message_id, reply_markup=get_main_menu_keyboard())

# --- Команды администратора ---
@bot.message_handler(commands=['add_user'])
def handle_add_user(message):
    if message.from_user.id != ADMIN_USER_ID:
        bot.reply_to(message, "Эта команда доступна только администратору бота.")
        return
    
    args = message.text.split(maxsplit=2)
    if len(args) != 3:
        bot.reply_to(message, "Использование: /add_user [user_id] [Имя_семьи]")
        return
    
    try:
        target_user_id = int(args[1])
        family_name = args[2].strip()
    except ValueError:
        bot.reply_to(message, "Неверный формат user_id.")
        return
    
    conn = get_db_connection()
    if not conn:
        bot.reply_to(message, "Проблема с подключением к базе данных.")
        return
    
    try:
        cursor = conn.cursor()
        
        cursor.execute("SELECT 1 FROM user_families WHERE user_id = %s", (target_user_id,))
        if cursor.fetchone():
            bot.reply_to(message, "Этот пользователь уже состоит в какой-то семье.")
            return

        invite_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        cursor.execute(
            "INSERT INTO families (name, owner_user_id, invite_code) VALUES (%s, %s, %s) RETURNING id",
            (family_name, target_user_id, invite_code)
        )
        family_id = cursor.fetchone()[0]
        
        create_expense_table(family_id, conn)

        cursor.execute(
            "INSERT INTO user_families (user_id, family_id, role) VALUES (%s, %s, 'admin')",
            (target_user_id, family_id)
        )
        conn.commit()
        bot.reply_to(message, f"Пользователь {target_user_id} успешно добавлен в семью '{family_name}' (ID: {family_id}).")
    except Exception as e:
        conn.rollback()
        bot.reply_to(message, f"Произошла ошибка при добавлении пользователя: {e}")
    finally:
        conn.close()

@bot.message_handler(commands=['get_invite_code'])
def handle_get_invite_code(message):
    if message.from_user.id != ADMIN_USER_ID:
        bot.reply_to(message, "Эта команда доступна только администратору бота.")
        return
    
    args = message.text.split()
    if len(args) != 2:
        bot.reply_to(message, "Использование: /get_invite_code [family_id]")
        return
    
    try:
        family_id = int(args[1])
    except ValueError:
        bot.reply_to(message, "Неверный формат family_id.")
        return
    
    conn = get_db_connection()
    if not conn:
        bot.reply_to(message, "Проблема с подключением к базе данных.")
        return
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT invite_code FROM families WHERE id = %s", (family_id,))
        invite_code = cursor.fetchone()
        
        if invite_code:
            bot.reply_to(message, f"Код-приглашение для семьи {family_id}: `{invite_code[0]}`", parse_mode='Markdown')
        else:
            bot.reply_to(message, f"Семья с ID {family_id} не найдена.")
    except Exception as e:
        bot.reply_to(message, f"Произошла ошибка: {e}")
    finally:
        conn.close()

@bot.message_handler(commands=['view_data'])
def handle_view_data(message):
    if message.from_user.id != ADMIN_USER_ID:
        bot.reply_to(message, "Эта команда доступна только администратору бота.")
        return
    
    conn = get_db_connection()
    if not conn:
        bot.reply_to(message, "Проблема с подключением к базе данных.")
        return
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM families")
        families = cursor.fetchall()
        
        family_info = "📊 **Информация по семьям:**\n\n"
        for family in families:
            family_id, family_name = family
            family_info += f"**Семья ID {family_id}:** {family_name}\n"
            
            cursor.execute("SELECT user_id, role FROM user_families WHERE family_id = %s", (family_id,))
            members = cursor.fetchall()
            family_info += "  - **Члены:**\n"
            for member in members:
                user_id, role = member
                try:
                    user_chat = bot.get_chat(user_id)
                    name = f"@{user_chat.username}" if user_chat.username else user_chat.first_name
                except Exception:
                    name = f"Пользователь {user_id}"
                family_info += f"    - {name} ({role})\n"
        
        bot.reply_to(message, family_info, parse_mode='Markdown')
    except Exception as e:
        bot.reply_to(message, f"Произошла ошибка при получении данных: {e}")
    finally:
        conn.close()

def parse_date_period(text):
    text_lower = text.lower()
    start_date = None
    end_date = datetime.now()
    if 'сегодня' in text_lower:
        start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    elif 'неделя' in text_lower:
        start_date = datetime.now() - timedelta(weeks=1)
        start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    elif 'месяц' in text_lower:
        start_date = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    elif 'год' in text_lower:
        start_date = datetime.now().replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
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


# --- Запуск бота ---
if __name__ == '__main__':
    train_model(TRAINING_DATA)
    init_db()
    
    commands = [
        telebot.types.BotCommand("/start", "Перезапустить бота"),
        telebot.types.BotCommand("/menu", "Показать главное меню"),
        telebot.types.BotCommand("/report", "Получить отчет о расходах"),
        telebot.types.BotCommand("/add_user", "Добавить пользователя (только админ)"),
        telebot.types.BotCommand("/get_invite_code", "Получить код (только админ)"),
        telebot.types.BotCommand("/view_data", "Просмотреть данные (только админ)"),
    ]
    bot.set_my_commands(commands)
    
    print("Бот запущен и готов принимать сообщения...")
    bot.polling(none_stop=True)