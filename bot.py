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

ADMIN_USER_ID = int(os.environ.get('ADMIN_USER_ID', '0'))

# --- Модель классификации (scikit-learn) ---
vectorizer = TfidfVectorizer()
classifier = LogisticRegression(max_iter=1000)

TRAINING_DATA = [
    ("хлеб", "Еда"), ("молоко", "Еда"), ("яйца", "Еда"), ("фрукты", "Еда"),
    ("овощи", "Еда"), ("продукты", "Еда"), ("обед", "Еда"), ("ужин", "Еда"),
    ("кофе", "Еда"), ("сок", "Еда"), ("чай", "Еда"), ("вода", "Еда"),
    ("булочка", "Еда"), ("пицца", "Еда"), ("рыба", "Еда"), ("мясо", "Еда"),
    ("колбаса", "Еда"), ("сыр", "Еда"), ("Еда"), ("картошка", "Еда"),
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
                owner_user_id BIGINT UNIQUE,
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
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recurring_payments (
                id SERIAL PRIMARY KEY,
                user_id BIGINT NOT NULL,
                family_id INTEGER NULL,
                title TEXT NOT NULL,
                amount REAL NULL,
                currency TEXT NULL,
                next_due_date DATE NOT NULL,
                recurrence_interval_unit TEXT NOT NULL,
                recurrence_interval_n INTEGER NOT NULL,
                reminder_offset_days INTEGER NOT NULL DEFAULT 7,
                last_reminded_date DATE NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
    cursor.execute("""
        SELECT family_id FROM user_families WHERE user_id = %s ORDER BY family_id LIMIT 1
    """, (user_id,))
    family_id = cursor.fetchone()
    conn.close()
    return family_id[0] if family_id else None

def is_family_subscription_active(family_id):
    if not family_id:
        return False
    conn = get_db_connection()
    if not conn: return False
    cursor = conn.cursor()
    cursor.execute(
        "SELECT subscription_end_date, is_active FROM families WHERE id = %s",
        (family_id,)
    )
    result = cursor.fetchone()
    conn.close()
    if result:
        end_date, active_flag = result
        return active_flag and end_date and end_date > datetime.now()
    return False

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
    conn = get_db_connection()
    if not conn:
        bot.reply_to(message, "Проблема с подключением к базе данных.", reply_markup=get_main_menu_keyboard())
        return
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM user_families WHERE user_id = %s", (user_id,))
    is_registered = cursor.fetchone()
    
    if not is_registered:
        try:
            family_name = f"Одиночный_{user_id}"
            invite_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
            cursor.execute(
                "INSERT INTO families (name, owner_user_id, invite_code) VALUES (%s, %s, %s) RETURNING id",
                (family_name, user_id, invite_code)
            )
            family_id = cursor.fetchone()[0]
            cursor.execute(
                "INSERT INTO user_families (user_id, family_id, role) VALUES (%s, %s, 'admin')",
                (user_id, family_id)
            )
            conn.commit()
            bot.reply_to(message, "Привет! Ты был автоматически зарегистрирован в одиночной семье. "
                                   "Теперь ты можешь добавлять расходы и смотреть отчеты. "
                                   "Если захочешь пригласить других, используй меню 'Семья'.", reply_markup=get_main_menu_keyboard())
        except psycopg2.errors.UniqueViolation:
            conn.rollback()
            bot.reply_to(message, "Вы уже зарегистрированы. Выбери опцию ниже:", reply_markup=get_main_menu_keyboard())
        except Exception as e:
            conn.rollback()
            bot.reply_to(message, f"Произошла ошибка при регистрации: {e}")
        finally:
            conn.close()
    else:
        conn.close()
        bot.reply_to(message, "С возвращением! Выбери опцию ниже:", reply_markup=get_main_menu_keyboard())

# --- Основные хэндлеры для меню ---
@bot.message_handler(regexp='^💸 Добавить расход$')
def handle_add_expense_menu(message):
    user_id = message.from_user.id
    family_id = get_user_active_family_id(user_id)
    if family_id is None:
        bot.send_message(message.chat.id, "Вы не состоите в активной семье. Для записи расходов вам нужно создать семью (админ) или присоединиться к существующей. Напишите /start.", reply_markup=get_main_menu_keyboard())
        return
    bot.send_message(message.chat.id, "Отлично! Просто напиши 'описание сумма валюта', например: 'хлеб 100тг'.", reply_markup=types.ReplyKeyboardRemove())

@bot.message_handler(regexp='^📊 Отчеты$')
def handle_report_menu(message):
    user_id = message.from_user.id
    family_id = get_user_active_family_id(user_id)
    if family_id is None:
        bot.send_message(message.chat.id, "Вы не состоите в активной семье. Для получения отчетов нужна семья.", reply_markup=get_main_menu_keyboard())
        return
    bot.send_message(message.chat.id, "За какой период вы хотите отчет?", reply_markup=get_report_period_keyboard())

@bot.message_handler(regexp='^⏰ Напоминания$')
def handle_reminders_menu(message):
    user_id = message.from_user.id
    family_id = get_user_active_family_id(user_id)
    if family_id is None:
        bot.send_message(message.chat.id, "Вы не состоите в активной семье. Для управления напоминаниями нужна семья.", reply_markup=get_main_menu_keyboard())
        return
    
    keyboard = types.InlineKeyboardMarkup(row_width=1)
    btn1 = types.InlineKeyboardButton("➕ Добавить напоминание", callback_data='add_recurring')
    btn2 = types.InlineKeyboardButton("📝 Посмотреть мои напоминания", callback_data='my_reminders')
    keyboard.add(btn1, btn2)
    bot.send_message(message.chat.id, "Что вы хотите сделать с напоминаниями?", reply_markup=keyboard)

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
    keyboard.add(btn1, btn2, btn3)
    bot.send_message(message.chat.id, "Что вы хотите сделать с семьей?", reply_markup=keyboard)

# --- Callback хэндлеры для кнопок ---
@bot.callback_query_handler(func=lambda call: call.data.startswith('report_'))
def handle_report_callback(call):
    chat_id = call.message.chat.id
    user_id = call.from_user.id
    period_text = call.data.replace('report_', '')
    
    bot.edit_message_reply_markup(chat_id, call.message.message_id) # Удаляем кнопки
    
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

@bot.callback_query_handler(func=lambda call: call.data == 'add_recurring')
def handle_add_recurring_callback(call):
    bot.edit_message_text("Отлично! Начнем с добавления напоминания. "
                          "Введите название регулярного платежа (например, 'Страховка машины'):",
                          call.message.chat.id, call.message.message_id, reply_markup=None)
    bot.register_next_step_handler(call.message, add_recurring_step2_title)

@bot.callback_query_handler(func=lambda call: call.data == 'my_reminders')
def handle_my_reminders_callback(call):
    bot.delete_message(call.message.chat.id, call.message.message_id)
    show_my_reminders(call.message)

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
        bot.edit_message_text("Вы успешно вышли из семьи. Чтобы снова добавлять расходы, вам нужно написать /start.", call.message.chat.id, call.message.message_id, reply_markup=get_onboarding_keyboard())
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
            # Для отображения имени, мы будем использовать метод get_chat. Это не очень эффективно
            # и может быть медленно, но для небольшой семьи это подойдет.
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
            SELECT f.id, f.name, f.invite_code, f.subscription_end_date
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
        family_id, family_name, invite_code, sub_end_date = family_info
        sub_status = "Активна" if sub_end_date and sub_end_date > datetime.now() else "Истекла"
        sub_date_str = sub_end_date.strftime('%Y-%m-%d') if sub_end_date else "Не установлена"
        
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
# Администратор теперь использует эту команду для добавления пользователей
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

# --- Логика добавления регулярного платежа (пошаговый) ---
def add_recurring_step2_title(message):
    chat_id = message.chat.id
    title = message.text.strip()
    if not title:
        bot.send_message(chat_id, "Название не может быть пустым. Отменил добавление.", reply_markup=get_main_menu_keyboard())
        return
    bot.send_message(chat_id, f"Название: '{title}'. Теперь введите сумму и валюту (например, '50000 тг' или 'Неважно'):")
    bot.register_next_step_handler(message, add_recurring_step3_amount, title)

def add_recurring_step3_amount(message, title):
    chat_id = message.chat.id
    amount_text = message.text.strip().lower()
    amount = None
    currency = None
    if amount_text != 'неважно':
        match = re.search(r'(\d+[\s\d.,]*)([а-яА-ЯёЁa-zA-Z$₽]{1,4})?', amount_text)
        if match:
            try:
                amount = float(match.group(1).replace(' ', '').replace(',', '.'))
                currency_part = match.group(2).strip().lower() if match.group(2) else 'тг'
                if currency_part in ['тг', 'kzt', 'тенге', '$', 'usd', 'руб', 'rub', '₽', 'eur']:
                    currency = currency_part
                else:
                    currency = 'тг'
            except ValueError:
                pass
        if amount is None:
            bot.send_message(chat_id, "Не могу распознать сумму. Пожалуйста, введите корректную сумму (например, '50000 тг') или 'Неважно'. Отменил добавление.", reply_markup=get_main_menu_keyboard())
            return
    bot.send_message(chat_id, f"Сумма: {amount_text}. Теперь введите дату следующего платежа (ГГГГ-ММ-ДД, например, '2026-07-15'):")
    bot.register_next_step_handler(message, add_recurring_step4_due_date, title, amount, currency)

def add_recurring_step4_due_date(message, title, amount, currency):
    chat_id = message.chat.id
    due_date_str = message.text.strip()
    try:
        next_due_date = datetime.strptime(due_date_str, '%Y-%m-%d').date()
    except ValueError:
        bot.send_message(chat_id, "Неверный формат даты. Используйте ГГГГ-ММ-ДД. Отменил добавление.", reply_markup=get_main_menu_keyboard())
        return
    bot.send_message(chat_id, f"Дата платежа: {next_due_date.strftime('%Y-%m-%d')}. Теперь введите периодичность (например, 'год', 'месяц', 'неделя' или 'день'):")
    bot.register_next_step_handler(message, add_recurring_step5_recurrence, title, amount, currency, next_due_date)

def add_recurring_step5_recurrence(message, title, amount, currency, next_due_date):
    chat_id = message.chat.id
    recurrence_text = message.text.strip().lower()
    recurrence_interval_unit = None
    recurrence_interval_n = 1
    match = re.search(r'(каждые\s*(\d+)\s*)?(.+)', recurrence_text)
    if match:
        n_str = match.group(2)
        if n_str: recurrence_interval_n = int(n_str)
        unit_text = match.group(3).strip()
        if 'год' in unit_text: recurrence_interval_unit = 'YEAR'
        elif 'месяц' in unit_text: recurrence_interval_unit = 'MONTH'
        elif 'недел' in unit_text: recurrence_interval_unit = 'WEEK'
        elif 'день' in unit_text: recurrence_interval_unit = 'DAY'
    if recurrence_interval_unit is None:
        bot.send_message(chat_id, "Не могу распознать периодичность. Попробуйте 'год', 'месяц', 'неделя', 'день' или 'каждые 2 месяца'. Отменил добавление.", reply_markup=get_main_menu_keyboard())
        return
    bot.send_message(chat_id, f"Периодичность: {message.text.strip()}. Теперь введите, за сколько дней до даты платежа напомнить (число, например, '7' для недели):")
    bot.register_next_step_handler(message, add_recurring_step6_reminder_offset, 
                                   title, amount, currency, next_due_date, 
                                   recurrence_interval_unit, recurrence_interval_n)

def add_recurring_step6_reminder_offset(message, title, amount, currency, next_due_date, recurrence_interval_unit, recurrence_interval_n):
    chat_id = message.chat.id
    reminder_offset_str = message.text.strip()
    try:
        reminder_offset_days = int(reminder_offset_str)
        if reminder_offset_days < 0: raise ValueError
    except ValueError:
        bot.send_message(chat_id, "Неверный формат числа дней. Отменил добавление.", reply_markup=get_main_menu_keyboard())
        return
    user_id = message.from_user.id
    family_id = get_user_active_family_id(user_id)
    if family_id is None:
        bot.send_message(chat_id, "Произошла ошибка: не удалось определить активную семью. Отменил добавление.", reply_markup=get_main_menu_keyboard())
        return
    conn = get_db_connection()
    if not conn:
        bot.send_message(chat_id, "Проблема с подключением к базе данных.", reply_markup=get_main_menu_keyboard())
        return
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO recurring_payments (user_id, family_id, title, amount, currency, next_due_date, 
                                            recurrence_interval_unit, recurrence_interval_n, reminder_offset_days)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (user_id, family_id, title, amount, currency, next_due_date, 
             recurrence_interval_unit, recurrence_interval_n, reminder_offset_days)
        )
        conn.commit()
        bot.send_message(chat_id, f"Регулярный платеж '{title}' добавлен! Напоминание придет за {reminder_offset_days} дней до {next_due_date.strftime('%Y-%m-%d')}.", reply_markup=get_main_menu_keyboard())
    except Exception as e:
        conn.rollback()
        bot.send_message(chat_id, f"Произошла ошибка при сохранении напоминания: {e}", reply_markup=get_main_menu_keyboard())
    finally:
        if conn: conn.close()

def show_my_reminders(message):
    user_id = message.from_user.id
    family_id = get_user_active_family_id(user_id)
    conn = get_db_connection()
    if not conn:
        bot.send_message(message.chat.id, "Проблема с подключением к базе данных.", reply_markup=get_main_menu_keyboard())
        return
    try:
        cursor = conn.cursor()
        if family_id:
            cursor.execute(
                """
                SELECT title, amount, currency, next_due_date, recurrence_interval_unit, recurrence_interval_n, reminder_offset_days
                FROM recurring_payments
                WHERE family_id = %s
                ORDER BY next_due_date ASC
                """,
                (family_id,)
            )
        else:
             cursor.execute(
                """
                SELECT title, amount, currency, next_due_date, recurrence_interval_unit, recurrence_interval_n, reminder_offset_days
                FROM recurring_payments
                WHERE user_id = %s AND family_id IS NULL
                ORDER BY next_due_date ASC
                """,
                (user_id,)
            )
        reminders = cursor.fetchall()
        if not reminders:
            bot.send_message(message.chat.id, "У вас пока нет регулярных платежей или напоминаний.", reply_markup=get_main_menu_keyboard())
            return
        response_text = "Ваши регулярные платежи и напоминания:\n\n"
        for r in reminders:
            title, amount, currency, next_due_date, unit, n, offset = r
            amount_str = f"{amount} {currency}" if amount else "Без суммы"
            recurrence_str = f"каждые {n} {unit.lower()}" if n > 1 else unit.lower()
            response_text += (f"🔹 **{title}**: Сумма: {amount_str}. Следующая дата: {next_due_date.strftime('%Y-%m-%d')}. "
                              f"Повторяется: {recurrence_str}. Напоминать за {offset} дней.\n\n")
        bot.send_message(message.chat.id, response_text, parse_mode='Markdown', reply_markup=get_main_menu_keyboard())
    except Exception as e:
        print(f"Ошибка при получении напоминаний: {e}")
        bot.send_message(message.chat.id, "Произошла ошибка при получении ваших напоминаний.", reply_markup=get_main_menu_keyboard())
    finally:
        if conn: conn.close()

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

# --- Хэндлер для текстовых сообщений (добавление расходов) ---
@bot.message_handler(func=lambda message: True, content_types=['text'])
def handle_text_messages(message):
    user_id = message.from_user.id
    family_id = get_user_active_family_id(user_id)
    
    if message.text in ['💸 Добавить расход', '📊 Отчеты', '⏰ Напоминания', '👨‍👩‍👧‍👦 Семья']:
        return

    if family_id is None:
        bot.send_message(message.chat.id, "Вы не состоите в активной семье. Для записи расходов вам нужно написать /start.", reply_markup=get_main_menu_keyboard())
        return

    text = message.text
    match = re.search(r'(.+)\s+([\d\s.,]+)\s*([а-яА-ЯёЁa-zA-Z$₽]{1,4})?', text)
    if match:
        description = match.group(1).strip()
        amount_str = match.group(2).strip().replace(' ', '').replace(',', '.')
        currency_part = match.group(3).strip().lower() if match.group(3) else 'тг'
        try:
            amount = float(amount_str)
            if currency_part in ['тг', 'kzt', 'тенге', '$', 'usd', 'руб', 'rub', '₽', 'eur']:
                currency = currency_part
            else:
                currency = 'тг'
            category = classify_expense(description)
            
            conn = get_db_connection()
            if conn:
                cursor = conn.cursor()
                try:
                    cursor.execute(
                        "INSERT INTO expenses (user_id, family_id, amount, currency, description, category) VALUES (%s, %s, %s, %s, %s, %s)",
                        (user_id, family_id, amount, currency, description, category)
                    )
                    conn.commit()
                    bot.send_message(message.chat.id, f"✅ Расход '{description}' ({amount} {currency}) успешно добавлен в категорию '{category}'.", reply_markup=get_main_menu_keyboard())
                except Exception as e:
                    conn.rollback()
                    bot.send_message(message.chat.id, f"Произошла ошибка при сохранении расхода: {e}", reply_markup=get_main_menu_keyboard())
                finally:
                    conn.close()
                return
            else:
                bot.send_message(message.chat.id, "Проблема с подключением к базе данных.", reply_markup=get_main_menu_keyboard())
                return
        except ValueError:
             bot.send_message(message.chat.id, "Не могу распознать сумму. Пожалуйста, используйте формат 'описание сумма валюта', например: 'хлеб 100тг'.", reply_markup=get_main_menu_keyboard())
             return

    bot.send_message(message.chat.id, "Не могу распознать расход. Пожалуйста, используйте формат 'описание сумма валюта', например: 'хлеб 100тг'.", reply_markup=get_main_menu_keyboard())

# --- Фоновая задача для напоминаний ---
def check_and_send_reminders():
    print("Запускаю проверку напоминаний...")
    conn = get_db_connection()
    if not conn:
        print("Не удалось подключиться к БД для проверки напоминаний.")
        return
    cursor = conn.cursor()
    today = date.today()
    try:
        cursor.execute(
            """
            SELECT id, user_id, family_id, title, next_due_date, recurrence_interval_unit, recurrence_interval_n, reminder_offset_days, last_reminded_date
            FROM recurring_payments
            """
        )
        reminders = cursor.fetchall()
        for r in reminders:
            rp_id, user_id, family_id, title, next_due_date, unit, n, offset, last_reminded_date = r
            days_until_due = (next_due_date - today).days
            if days_until_due == offset:
                if not last_reminded_date or last_reminded_date != today:
                    message_text = (
                        f"🔔 Напоминание: **'{title}'**\n"
                        f"Следующий платеж: {next_due_date.strftime('%d.%m.%Y')}\n"
                        f"До него осталось {offset} дней."
                    )
                    try:
                        if family_id:
                            cursor.execute("SELECT user_id FROM user_families WHERE family_id = %s", (family_id,))
                            family_members = cursor.fetchall()
                            for member_id in family_members:
                                bot.send_message(member_id[0], message_text, parse_mode='Markdown')
                        else:
                            bot.send_message(user_id, message_text, parse_mode='Markdown')
                        cursor.execute(
                            "UPDATE recurring_payments SET last_reminded_date = %s WHERE id = %s",
                            (today, rp_id)
                        )
                        conn.commit()
                        print(f"Отправлено напоминание для {title} (ID: {rp_id})")
                    except Exception as send_e:
                        print(f"Ошибка при отправке напоминания для {title} (ID: {rp_id}): {send_e}")
                        conn.rollback()
            if next_due_date <= today and last_reminded_date == today:
                new_due_date = next_due_date
                if unit == 'YEAR': new_due_date = new_due_date.replace(year=new_due_date.year + n)
                elif unit == 'MONTH':
                    new_month = new_due_date.month + n
                    new_year = new_due_date.year + (new_month - 1) // 12
                    new_month = (new_month - 1) % 12 + 1
                    new_due_date = new_due_date.replace(year=new_year, month=new_month)
                elif unit == 'WEEK': new_due_date = new_due_date + timedelta(weeks=n)
                elif unit == 'DAY': new_due_date = new_due_date + timedelta(days=n)
                if new_due_date > next_due_date:
                    cursor.execute(
                        "UPDATE recurring_payments SET next_due_date = %s, last_reminded_date = NULL WHERE id = %s",
                        (new_due_date, rp_id)
                    )
                    conn.commit()
                    print(f"Дата следующего платежа для '{title}' сдвинута на {new_due_date.strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"Критическая ошибка при проверке напоминаний: {e}")
    finally:
        if conn: conn.close()

# --- Настройка постоянного меню ---
def set_commands():
    commands = [
        telebot.types.BotCommand("/start", "Перезапустить бота"),
        telebot.types.BotCommand("/menu", "Показать главное меню"),
        telebot.types.BotCommand("/report", "Получить отчет о расходах"),
        telebot.types.BotCommand("/add_recurring", "Добавить регулярный платеж"),
        telebot.types.BotCommand("/my_reminders", "Посмотреть мои напоминания")
    ]
    bot.set_my_commands(commands)
    print("Команды бота установлены.")

# --- Запуск бота и фоновой задачи ---
if __name__ == '__main__':
    train_model(TRAINING_DATA)
    init_db()
    set_commands()
    
    # Для Railway Cron Job, этот блок должен быть в отдельном файле (reminder_worker.py)
    # import threading
    # reminder_thread = threading.Thread(target=check_and_send_reminders)
    # reminder_thread.daemon = True
    # reminder_thread.start()

    print("Бот запущен и готов принимать сообщения...")
    bot.polling(none_stop=True)