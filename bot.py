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
import time
import random
import string
from dotenv import load_dotenv

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

# Используйте ваш ID Telegram в качестве ID администратора
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
            CREATE TABLE IF NOT EXISTS expenses (
                id SERIAL PRIMARY KEY,
                user_id BIGINT,
                family_id INTEGER NULL,
                amount REAL,
                currency TEXT,
                description TEXT,
                category TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        ''')
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
        conn.close()

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
        conn.close()

# --- UI (User Interface) ---
def get_main_menu_keyboard():
    keyboard = types.ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    btn_add_expense = types.KeyboardButton('💸 Добавить расход')
    btn_report = types.KeyboardButton('📊 Отчеты')
    btn_add_recurring = types.KeyboardButton('⏰ Напоминания')
    btn_manage_family = types.KeyboardButton('👨‍👩‍👧‍👦 Семья')
    keyboard.add(btn_add_expense, btn_report, btn_add_recurring, btn_manage_family)
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
    
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT category, SUM(amount)
            FROM expenses
            WHERE family_id = %s AND timestamp BETWEEN %s AND %s
            GROUP BY category
        ''', (family_id, start_date, end_date))
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
            SELECT uf.user_id, uf.role
            FROM user_families uf
            WHERE uf.family_id = %s
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
@bot.message_handler(commands=['add_member'])
def handle_add_member(message):
    if message.from_user.id != ADMIN_USER_ID:
        bot.reply_to(message, "Эта команда доступна только администратору бота.")
        return
    
    args = message.text.split()
    if len(args) != 3:
        bot.reply_to(message, "Использование: /add_member [user_id] [family_id]")
        return
    
    try:
        target_user_id = int(args[1])
        family_id = int(args[2])
    except ValueError:
        bot.reply_to(message, "Неверный формат user_id или family_id.")
        return
    
    conn = get_db_connection()
    if not conn:
        bot.reply_to(message, "Проблема с подключением к базе данных.")
        return
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM user_families WHERE family_id = %s", (family_id,))
        member_count = cursor.fetchone()[0]
        if member_count >= 5:
            bot.reply_to(message, "Нельзя добавить больше 5 человек в семью.")
            return

        cursor.execute("SELECT 1 FROM user_families WHERE user_id = %s", (target_user_id,))
        if cursor.fetchone():
            bot.reply_to(message, "Этот пользователь уже состоит в какой-то семье.")
            return

        cursor.execute(
            "INSERT INTO user_families (user_id, family_id, role) VALUES (%s, %s, 'member')",
            (target_user_id, family_id)
        )
        conn.commit()
        bot.reply_to(message, f"Пользователь {target_user_id} успешно добавлен в семью {family_id}.")
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

@bot.message_handler(commands=['create_family'])
def handle_create_family(message):
    if message.from_user.id != ADMIN_USER_ID:
        bot.reply_to(message, "Эта команда доступна только администратору бота.")
        return
    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        bot.reply_to(message, "Использование: /create_family [Название_семьи]")
        return
    family_name = args[1].strip()
    conn = get_db_connection()
    if not conn:
        bot.reply_to(message, "Проблема с подключением к базе данных.")
        return
    try:
        cursor = conn.cursor()
        invite_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        cursor.execute(
            "INSERT INTO families (name, owner_user_id, invite_code) VALUES (%s, %s, %s) RETURNING id",
            (family_name, message.from_user.id, invite_code)
        )
        family_id = cursor.fetchone()[0]
        bot.reply_to(message, f"Семья '{family_name}' (ID: {family_id}) создана. Код приглашения: `{invite_code}`.", parse_mode='Markdown')
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        bot.reply_to(message, "Семья с таким названием уже существует.")
    except Exception as e:
        conn.rollback()
        bot.reply_to(message, f"Произошла ошибка при создании семьи: {e}")
    finally:
        conn.close()

@bot.message_handler(commands=['set_subscription'])
def handle_set_subscription(message):
    if message.from_user.id != ADMIN_USER_ID:
        bot.reply_to(message, "Эта команда доступна только администратору бота.")
        return
    args = message.text.split()
    if len(args) != 3:
        bot.reply_to(message, "Использование: /set_subscription [ID_семьи] [ГГГГ-ММ-ДД]")
        return
    try:
        family_id = int(args[1])
        end_date_str = args[2]
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    except ValueError:
        bot.reply_to(message, "Неверный формат ID семьи или даты. Используйте: /set_subscription [ID_семьи] [ГГГГ-ММ-ДД]")
        return
    conn = get_db_connection()
    if not conn:
        bot.reply_to(message, "Проблема с подключением к базе данных.")
        return
    try:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE families SET subscription_end_date = %s, is_active = TRUE WHERE id = %s",
            (end_date, family_id)
        )
        conn.commit()
        if cursor.rowcount > 0:
            bot.reply_to(message, f"Подписка для семьи ID {family_id} установлена до {end_date.strftime('%Y-%m-%d')}.")
        else:
            bot.reply_to(message, f"Семья с ID {family_id} не найдена.")
    except Exception as e:
        conn.rollback()
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

# --- Обработка прямых инвайт-кодов
def handle_invite_code_direct(message):
    invite_code = message.text.strip()
    user_id = message.from_user.id
    
    conn = get_db_connection()
    if not conn:
        bot.send_message(message.chat.id, "Проблема с подключением к базе данных.", reply_markup=get_main_menu_keyboard())
        return
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM user_families WHERE user_id = %s", (user_id,))
        if cursor.fetchone():
            bot.send_message(message.chat.id, "Вы уже состоите в семье. Чтобы присоединиться к новой, нужно сначала выйти из текущей.", reply_markup=get_main_menu_keyboard())
            return
        
        cursor.execute("SELECT id, name FROM families WHERE invite_code = %s", (invite_code,))
        family_info = cursor.fetchone()
        
        if family_info:
            family_id, family_name = family_info
            cursor.execute("SELECT COUNT(*) FROM user_families WHERE family_id = %s", (family_id,))
            member_count = cursor.fetchone()[0]
            if member_count >= 5:
                bot.send_message(message.chat.id, "Нельзя добавить больше 5 человек в эту семью.", reply_markup=get_main_menu_keyboard())
                return

            cursor.execute(
                "INSERT INTO user_families (user_id, family_id, role) VALUES (%s, %s, 'member')",
                (user_id, family_id)
            )
            conn.commit()
            bot.send_message(message.chat.id, f"Добро пожаловать в семью '{family_name}'! 🎉", reply_markup=get_main_menu_keyboard())
        else:
            bot.send_message(message.chat.id, "❌ Неверный код приглашения или такой семьи не существует. Попробуйте еще раз.", reply_markup=get_main_menu_keyboard())
    except Exception as e:
        conn.rollback()
        bot.send_message(message.chat.id, f"Произошла ошибка при присоединении к семье: {e}", reply_markup=get_main_menu_keyboard())
    finally:
        conn.close()

# --- Хэндлер для текстовых сообщений (добавление расходов) ---
@bot.message_handler(func=lambda message: True, content_types=['text'])
def handle_text_messages(message):
    user_id = message.from_user.id
    family_id = get_user_active_family_id(user_id)
    
    # Обработка нажатий на кнопки
    if message.text in ['💸 Добавить расход', '📊 Отчеты', '👨‍👩‍👧‍👦 Семья']:
        return
    
    # Обработка инвайт-кодов, отправленных напрямую
    if len(message.text) == 8 and message.text.isalnum():
        handle_invite_code_direct(message)
        return

    if family_id is None:
        bot.send_message(message.chat.id, "Вы не состоите в активной семье. Для записи расходов вам нужно быть добавленным администратором. Напишите /start.", reply_markup=get_main_menu_keyboard())
        return

    text = message.text
    # Новый парсер, который ищет несколько пар "описание сумма"
    pattern = r'([\w\s]+)\s+([\d\s.,]+)'
    matches = re.findall(pattern, text)
    
    if not matches:
        bot.send_message(message.chat.id, "Не могу распознать расход. Пожалуйста, используйте формат 'описание сумма', например: 'хлеб 100, молоко 500'.", reply_markup=get_main_menu_keyboard())
        return
    
    conn = get_db_connection()
    if not conn:
        bot.send_message(message.chat.id, "Проблема с подключением к базе данных.", reply_markup=get_main_menu_keyboard())
        return

    try:
        for match in matches:
            description = match[0].strip()
            amount_str = match[1].strip().replace(' ', '').replace(',', '.')
            
            try:
                amount = float(amount_str)
                currency = 'тг' # Валюта по умолчанию, можно изменить
                category = classify_expense(description)
                
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO expenses (user_id, family_id, amount, currency, description, category) VALUES (%s, %s, %s, %s, %s, %s)",
                    (user_id, family_id, amount, currency, description, category)
                )
            except ValueError:
                bot.send_message(message.chat.id, f"Не могу распознать сумму для '{description}'. Проверьте формат.", reply_markup=get_main_menu_keyboard())
                conn.rollback()
                return

        conn.commit()
        bot.send_message(message.chat.id, f"✅ Расходы успешно добавлены. Спасибо!", reply_markup=get_main_menu_keyboard())
    except Exception as e:
        conn.rollback()
        bot.send_message(message.chat.id, f"Произошла ошибка при сохранении расхода: {e}", reply_markup=get_main_menu_keyboard())
    finally:
        conn.close()


# --- Запуск бота ---
if __name__ == '__main__':
    train_model(TRAINING_DATA)
    init_db()
    
    # Настройка постоянного меню (по вашей идее)
    commands = [
        telebot.types.BotCommand("/start", "Перезапустить бота"),
        telebot.types.BotCommand("/menu", "Показать главное меню"),
        telebot.types.BotCommand("/report", "Получить отчет о расходах"),
        telebot.types.BotCommand("/add_member", "Добавить участника (только админ)"),
        telebot.types.BotCommand("/create_family", "Создать семью (только админ)"),
        telebot.types.BotCommand("/set_subscription", "Установить подписку (только админ)"),
        telebot.types.BotCommand("/view_data", "Просмотреть данные (только админ)"),
    ]
    bot.set_my_commands(commands)
    
    print("Бот запущен и готов принимать сообщения...")
    bot.polling(none_stop=True)