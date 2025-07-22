import telebot
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
import time # Для сна в фоновой задаче, если используем Worker

# --- Настройки бота ---
# BOT_TOKEN будет взят из переменных окружения Railway.
# НЕ вставляйте его здесь напрямую при загрузке на GitHub!
BOT_TOKEN = os.environ.get('BOT_TOKEN') 
if not BOT_TOKEN:
    print("Ошибка: Токен бота не найден. Установите переменную окружения BOT_TOKEN.")
    exit() # Бот не сможет работать без токена
bot = telebot.TeleBot(BOT_TOKEN)

# --- Настройки базы данных ---
# DATABASE_URL будет взят из переменных окружения Railway.
# НЕ вставляйте его здесь напрямую при загрузке на GitHub!
DATABASE_URL = os.environ.get('DATABASE_URL')
if not DATABASE_URL:
    print("Ошибка: URL базы данных не найден. Установите переменную окружения DATABASE_URL.")
    exit() # Бот не сможет работать без подключения к БД

# --- Модель классификации (scikit-learn) ---
vectorizer = TfidfVectorizer()
classifier = LogisticRegression(max_iter=1000)

# Простой тренировочный набор данных для обучения модели.
# Расширяйте его по мере использования для улучшения точности!
TRAINING_DATA = [
    ("хлеб", "Еда"), ("молоко", "Еда"), ("яйца", "Еда"), ("фрукты", "Еда"),
    ("овощи", "Еда"), ("продукты", "Еда"), ("обед", "Еда"), ("ужин", "Еда"),
    ("кофе", "Еда"), ("сок", "Еда"), ("чай", "Еда"), ("вода", "Еда"),
    ("булочка", "Еда"), ("пицца", "Еда"), ("рыба", "Еда"), ("мясо", "Еда"),
    
    ("бензин", "Транспорт"), ("такси", "Транспорт"), ("автобус", "Транспорт"),
    ("метро", "Транспорт"), ("проезд", "Транспорт"), ("поезд", "Транспорт"),
    ("самолет", "Транспорт"), ("маршрутка", "Транспорт"), ("проездной", "Транспорт"),

    ("билеты", "Развлечения"), ("кино", "Развлечения"), ("театр", "Развлечения"),
    ("концерт", "Развлечения"), ("книга", "Развлечения"), ("игры", "Развлечения"),
    ("аттракционы", "Развлечения"), ("музей", "Развлечения"), ("подписка", "Развлечения"),

    ("одежда", "Одежда"), ("обувь", "Одежда"), ("футболка", "Одежда"),
    ("брюки", "Одежда"), ("платье", "Одежда"), ("куртка", "Одежда"),

    ("коммуналка", "Жилье"), ("аренда", "Жилье"), ("свет", "Жилье"),
    ("вода", "Жилье"), ("газ", "Жилье"), ("квитанция", "Жилье"), ("ипотека", "Жилье"),

    ("аптека", "Здоровье"), ("врач", "Здоровье"), ("лекарства", "Здоровье"),
    ("стоматолог", "Здоровье"), ("витамины", "Здоровье"),

    ("связь", "Связь"), ("интернет", "Связь"), ("телефон", "Связь"),
    ("мобильный", "Связь"),

    ("скотч", "Дом/Канцелярия"), ("ручки", "Дом/Канцелярия"), ("бумага", "Дом/Канцелярия"),
    ("канцелярия", "Дом/Канцелярия"), ("дом", "Дом/Канцелярия"), ("посуда", "Дом/Канцелярия"),
    ("чистящие", "Дом/Канцелярия"), ("инструменты", "Дом/Канцелярия"),

    ("подарок", "Прочее"), ("другое", "Прочее"), ("разное", "Прочее"), 
    ("сюрприз", "Прочее"), ("налоги", "Прочее"), ("штраф", "Прочее")
]

# Функция для обучения/дообучения модели классификации
def train_model(data):
    if not data:
        print("Нет данных для обучения модели. Модель не будет обучена.")
        return

    descriptions = [item[0] for item in data]
    categories = [item[1] for item in data]

    # Преобразование текста в числовой формат (TF-IDF)
    X = vectorizer.fit_transform(descriptions)
    # Обучение классификатора
    classifier.fit(X, categories)
    print("Модель классификации успешно обучена.")

# Классификация расхода
def classify_expense(description):
    try:
        # Проверим, обучена ли модель (есть ли у нее классы)
        if not hasattr(classifier, 'classes_') or len(classifier.classes_) == 0:
            print("Модель еще не обучена или нет классов. Возвращаю 'Прочее'.")
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
        # Таблица для расходов
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS expenses (
                id SERIAL PRIMARY KEY,
                user_id BIGINT,
                family_id INTEGER NULL, -- Добавлено для семейных расходов
                amount REAL,
                currency TEXT,
                description TEXT,
                category TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        ''')
        # Таблица для семей/групп (если планируете семейные подписки)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS families (
                id SERIAL PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                owner_user_id BIGINT UNIQUE, -- Создатель семьи
                subscription_end_date TIMESTAMP NULL, -- Дата окончания подписки
                is_active BOOLEAN DEFAULT FALSE,
                invite_code TEXT UNIQUE NULL -- Код для присоединения
            );
        ''')
        # Таблица для связи пользователей и семей
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_families (
                user_id BIGINT NOT NULL,
                family_id INTEGER NOT NULL REFERENCES families(id) ON DELETE CASCADE,
                role TEXT DEFAULT 'member', -- 'admin', 'member'
                PRIMARY KEY (user_id, family_id)
            );
        ''')
        # Таблица для регулярных платежей/напоминаний
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recurring_payments (
                id SERIAL PRIMARY KEY,
                user_id BIGINT NOT NULL, -- Кто создал напоминание
                family_id INTEGER NULL, -- К какой семье относится, если не личному пользователю
                title TEXT NOT NULL,
                amount REAL NULL,
                currency TEXT NULL,
                next_due_date DATE NOT NULL, -- Следующая дата платежа
                recurrence_interval_unit TEXT NOT NULL, -- 'YEAR', 'MONTH', 'WEEK', 'DAY'
                recurrence_interval_n INTEGER NOT NULL, -- 1 для "каждый год"
                reminder_offset_days INTEGER NOT NULL DEFAULT 7, -- За сколько дней напомнить
                last_reminded_date DATE NULL, -- Дата последнего отправленного напоминания
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        ''')
        conn.commit()
        conn.close()
        print("База данных инициализирована (таблицы проверены/созданы).")

# --- Вспомогательные функции ---
# Получить активную семью пользователя (для упрощения примера)
# В реальном проекте пользователь может выбрать активную семью.
def get_user_active_family_id(user_id):
    conn = get_db_connection()
    if not conn: return None
    cursor = conn.cursor()
    # Пока что, просто проверим, является ли пользователь владельцем какой-либо семьи.
    # В более сложной системе здесь будет логика выбора/текущей активной семьи.
    cursor.execute("SELECT id FROM families WHERE owner_user_id = %s LIMIT 1", (user_id,))
    family_id = cursor.fetchone()
    conn.close()
    return family_id[0] if family_id else None

# Проверка подписки семьи
def is_family_subscription_active(family_id):
    if not family_id: # Если нет семьи, подписка неактивна
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
        # Подписка активна, если флаг True и дата окончания в будущем
        return active_flag and end_date and end_date > datetime.now()
    return False

# --- Команды бота ---
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Я твой помощник по учету расходов и напоминаний. "
                           "Чтобы добавить расход, просто напиши 'описание сумма валюта', "
                           "например: 'хлеб 100тг' или '1500тг бензин'.\n\n"
                           "Доступные команды:\n"
                           "/report - Получить отчет о расходах\n"
                           "/add_recurring - Добавить регулярный платеж\n"
                           "/my_reminders - Посмотреть мои напоминания\n"
                           "//set_subscription [ID_СЕМЬИ] [ГГГГ-ММ-ДД] - (ТОЛЬКО ДЛЯ АДМИНА) Установить подписку\n"
                           "//create_family [Имя_семьи] - (ТОЛЬКО ДЛЯ АДМИНА) Создать семью"
                           )

# --- Обработчик создания семьи (ТОЛЬКО ДЛЯ АДМИНА) ---
# Замените 'ВАШ_TELEGRAM_ID_АДМИНА' на ваш реальный ID в Telegram.
# Узнать свой ID можно через бота @userinfobot
ADMIN_USER_ID = int(os.environ.get('ADMIN_USER_ID', '0')) # Возьмите из переменных окружения

@bot.message_handler(commands=['create_family'])
def handle_create_family(message):
    if message.from_user.id != ADMIN_USER_ID:
        bot.reply_to(message, "Эта команда доступна только администратору бота.")
        return
    
    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        bot.reply_to(message, "Использование: //create_family [Название_семьи]")
        return
    
    family_name = args[1].strip()
    conn = get_db_connection()
    if not conn:
        bot.reply_to(message, "Проблема с подключением к базе данных.")
        return
    try:
        cursor = conn.cursor()
        # Генерируем простой инвайт-код
        invite_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        cursor.execute(
            "INSERT INTO families (name, owner_user_id, invite_code) VALUES (%s, %s, %s) RETURNING id",
            (family_name, message.from_user.id, invite_code)
        )
        family_id = cursor.fetchone()[0]
        # Делаем владельца также членом user_families
        cursor.execute(
            "INSERT INTO user_families (user_id, family_id, role) VALUES (%s, %s, 'admin')",
            (message.from_user.id, family_id)
        )
        conn.commit()
        bot.reply_to(message, f"Семья '{family_name}' (ID: {family_id}) создана. Код приглашения: `{invite_code}`. "
                               f"Вы стали её администратором.")
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        bot.reply_to(message, "Семья с таким названием или вы уже являетесь владельцем другой семьи.")
    except Exception as e:
        conn.rollback()
        bot.reply_to(message, f"Произошла ошибка при создании семьи: {e}")
    finally:
        conn.close()

# --- Обработчик установки подписки (ТОЛЬКО ДЛЯ АДМИНА) ---
import random
import string

@bot.message_handler(commands=['set_subscription'])
def handle_set_subscription(message):
    if message.from_user.id != ADMIN_USER_ID:
        bot.reply_to(message, "Эта команда доступна только администратору бота.")
        return
    
    args = message.text.split()
    if len(args) != 3:
        bot.reply_to(message, "Использование: //set_subscription [ID_семьи] [ГГГГ-ММ-ДД]")
        return
    
    try:
        family_id = int(args[1])
        end_date_str = args[2]
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    except ValueError:
        bot.reply_to(message, "Неверный формат ID семьи или даты. Используйте: //set_subscription [ID_семьи] [ГГГГ-ММ-ДД]")
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
            # Опционально: уведомить владельца семьи
            # cursor.execute("SELECT owner_user_id FROM families WHERE id = %s", (family_id,))
            # owner_id = cursor.fetchone()
            # if owner_id:
            #     bot.send_message(owner_id[0], f"Ваша семейная подписка активирована до {end_date.strftime('%Y-%m-%d')}!")
        else:
            bot.reply_to(message, f"Семья с ID {family_id} не найдена.")
    except Exception as e:
        conn.rollback()
        bot.reply_to(message, f"Произошла ошибка: {e}")
    finally:
        conn.close()

# --- Обработчик присоединения к семье ---
@bot.message_handler(commands=['join_family'])
def handle_join_family(message):
    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        bot.reply_to(message, "Использование: /join_family [Код_приглашения]")
        return
    
    invite_code = args[1].strip()
    user_id = message.from_user.id
    
    conn = get_db_connection()
    if not conn:
        bot.reply_to(message, "Проблема с подключением к базе данных.")
        return
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM families WHERE invite_code = %s", (invite_code,))
        family_info = cursor.fetchone()

        if family_info:
            family_id, family_name = family_info
            # Проверить, не является ли пользователь уже членом этой семьи
            cursor.execute("SELECT 1 FROM user_families WHERE user_id = %s AND family_id = %s", (user_id, family_id))
            if cursor.fetchone():
                bot.reply_to(message, f"Вы уже являетесь членом семьи '{family_name}'.")
            else:
                cursor.execute(
                    "INSERT INTO user_families (user_id, family_id, role) VALUES (%s, %s, 'member')",
                    (user_id, family_id)
                )
                conn.commit()
                bot.reply_to(message, f"Вы успешно присоединились к семье '{family_name}'.")
        else:
            bot.reply_to(message, "Неверный код приглашения.")
    except Exception as e:
        conn.rollback()
        bot.reply_to(message, f"Произошла ошибка при присоединении к семье: {e}")
    finally:
        conn.close()

# --- Обработчик добавления регулярного платежа ---
@bot.message_handler(commands=['add_recurring'])
def add_recurring_step1(message):
    user_id = message.from_user.id
    family_id = get_user_active_family_id(user_id) # Или запрос, чтобы пользователь выбрал семью
    # Если бот будет для одиночных пользователей, можно убрать family_id
    
    if family_id is None:
        bot.reply_to(message, "Вы не состоите в активной семье. Для добавления регулярных платежей нужна семья. Используйте /create_family (админ) или /join_family.")
        return

    # Проверка подписки перед добавлением регулярного платежа
    if not is_family_subscription_active(family_id):
        bot.reply_to(message, "Ваша подписка истекла. Пожалуйста, продлите ее для добавления регулярных платежей.")
        return

    bot.reply_to(message, "Введите название регулярного платежа (например, 'Страховка машины'):")
    bot.register_next_step_handler(message, add_recurring_step2_title)

def add_recurring_step2_title(message):
    chat_id = message.chat.id
    title = message.text.strip()
    if not title:
        bot.send_message(chat_id, "Название не может быть пустым. Отменил добавление.")
        return
    
    bot.send_message(chat_id, f"Название: '{title}'. Теперь введите сумму и валюту (например, '50000 тг' или 'Неважно'):")
    bot.register_next_step_handler(message, add_recurring_step3_amount, title)

def add_recurring_step3_amount(message, title):
    chat_id = message.chat.id
    amount_text = message.text.strip().lower()
    amount = None
    currency = None

    if amount_text != 'неважно':
        match = re.search(r'(\d+[\s\d.,]*)([а-яА-ЯёЁa-zA-Z$]{1,4})?', amount_text)
        if match:
            try:
                amount = float(match.group(1).replace(' ', '').replace(',', '.'))
                currency_part = match.group(2).strip().lower() if match.group(2) else 'тг'
                if currency_part in ['тг', 'kzt', 'тенге', '$', 'usd', 'руб', 'rub', '₽', 'eur']:
                    currency = currency_part
                elif len(currency_part) <= 4 and currency_part.isalpha():
                    currency = currency_part
                else:
                    currency = 'тг' # Дефолт
            except ValueError:
                pass
        
        if amount is None:
            bot.send_message(chat_id, "Не могу распознать сумму. Пожалуйста, введите корректную сумму (например, '50000 тг') или 'Неважно'. Отменил добавление.")
            return

    bot.send_message(chat_id, f"Сумма: {amount_text}. Теперь введите дату следующего платежа (ГГГГ-ММ-ДД, например, '2026-07-15'):")
    bot.register_next_step_handler(message, add_recurring_step4_due_date, title, amount, currency)

def add_recurring_step4_due_date(message, title, amount, currency):
    chat_id = message.chat.id
    due_date_str = message.text.strip()
    try:
        next_due_date = datetime.strptime(due_date_str, '%Y-%m-%d').date()
    except ValueError:
        bot.send_message(chat_id, "Неверный формат даты. Используйте ГГГГ-ММ-ДД. Отменил добавление.")
        return
    
    bot.send_message(chat_id, f"Дата платежа: {next_due_date.strftime('%Y-%m-%d')}. Теперь введите периодичность (например, 'год', 'месяц', 'неделя', 'день' или 'каждые 2 месяца'):")
    bot.register_next_step_handler(message, add_recurring_step5_recurrence, title, amount, currency, next_due_date)

def add_recurring_step5_recurrence(message, title, amount, currency, next_due_date):
    chat_id = message.chat.id
    recurrence_text = message.text.strip().lower()
    
    recurrence_interval_unit = None
    recurrence_interval_n = 1

    # Парсинг периодичности
    match = re.search(r'(каждые\s*(\d+)\s*)?(.+)', recurrence_text)
    if match:
        n_str = match.group(2)
        if n_str:
            recurrence_interval_n = int(n_str)
        unit_text = match.group(3).strip()

        if 'год' in unit_text: recurrence_interval_unit = 'YEAR'
        elif 'месяц' in unit_text: recurrence_interval_unit = 'MONTH'
        elif 'недел' in unit_text: recurrence_interval_unit = 'WEEK'
        elif 'день' in unit_text: recurrence_interval_unit = 'DAY'
    
    if recurrence_interval_unit is None:
        bot.send_message(chat_id, "Не могу распознать периодичность. Попробуйте 'год', 'месяц', 'неделя', 'день' или 'каждые 2 месяца'. Отменил добавление.")
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
        if reminder_offset_days < 0:
            raise ValueError
    except ValueError:
        bot.send_message(chat_id, "Неверный формат числа дней. Отменил добавление.")
        return
    
    user_id = message.from_user.id
    family_id = get_user_active_family_id(user_id) # Заново получаем family_id
    if family_id is None: # Повторная проверка на всякий случай
        bot.send_message(chat_id, "Произошла ошибка: не удалось определить активную семью. Отменил добавление.")
        return

    conn = get_db_connection()
    if not conn:
        bot.send_message(chat_id, "Проблема с подключением к базе данных.")
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
        bot.send_message(chat_id, f"Регулярный платеж '{title}' добавлен! Напоминание придет за {reminder_offset_days} дней до {next_due_date.strftime('%Y-%m-%d')}.")
    except Exception as e:
        conn.rollback()
        bot.send_message(chat_id, f"Произошла ошибка при сохранении напоминания: {e}")
    finally:
        conn.close()

# --- Обработчик просмотра напоминаний ---
@bot.message_handler(commands=['my_reminders'])
def show_my_reminders(message):
    user_id = message.from_user.id
    family_id = get_user_active_family_id(user_id)

    conn = get_db_connection()
    if not conn:
        bot.send_message(chat_id, "Проблема с подключением к базе данных.")
        return
    
    try:
        cursor = conn.cursor()
        if family_id:
            cursor.execute(
                """
                SELECT title, amount, currency, next_due_date, recurrence_interval_unit, recurrence_interval_n, reminder_offset_days
                FROM recurring_payments
                WHERE (user_id = %s OR family_id = %s)
                ORDER BY next_due_date ASC
                """,
                (user_id, family_id)
            )
        else: # Только личные напоминания, если нет семьи
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
            bot.send_message(message.chat.id, "У вас пока нет регулярных платежей или напоминаний.")
            return

        response_text = "Ваши регулярные платежи и напоминания:\n\n"
        for r in reminders:
            title, amount, currency, next_due_date, unit, n, offset = r
            amount_str = f"{amount} {currency}" if amount else "Без суммы"
            recurrence_str = f"каждые {n} {unit.lower()}" if n > 1 else unit.lower()
            response_text += (f"🔹 **{title}**: Сумма: {amount_str}. Следующая дата: {next_due_date.strftime('%Y-%m-%d')}. "
                              f"Повторяется: {recurrence_str}. Напоминать за {offset} дней.\n\n")
        
        bot.send_message(message.chat.id, response_text, parse_mode='Markdown')

    except Exception as e:
        print(f"Ошибка при получении напоминаний: {e}")
        bot.send_message(message.chat.id, "Произошла ошибка при получении ваших напоминаний.")
    finally:
        conn.close()

# --- Обработчик отчетов ---
@bot.message_handler(commands=['report'])
def handle_report_request(message):
    user_id = message.from_user.id
    family_id = get_user_active_family_id(user_id)
    if family_id is None: # Если нет семьи, подписка неактивна
        bot.reply_to(message, "Вы не состоите в активной семье. Для получения отчетов нужна семья.")
        return
    
    # Проверка подписки перед выдачей отчета
    if not is_family_subscription_active(family_id):
        bot.reply_to(message, "Ваша подписка истекла. Пожалуйста, продлите ее для получения отчетов.")
        return

    bot.reply_to(message, "За какой период вы хотите отчет? Например: 'сегодня', 'неделя', 'месяц', 'май 2025', 'с 01.01.2024 по 31.01.2024'.")
    bot.register_next_step_handler(message, process_report_period)

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
                    except ValueError:
                        pass
                
                for fmt in ["%d.%m.%Y", "%d.%m.%y", "%d.%m"]:
                    try:
                        end_date = datetime.strptime(date_to_str, fmt)
                        if fmt == "%d.%m": end_date = end_date.replace(year=datetime.now().year)
                        end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)
                        break
                    except ValueError:
                        pass
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
                    if month_num == 12:
                        end_date = datetime(year + 1, 1, 1, 0, 0, 0, 0) - timedelta(microseconds=1)
                    else:
                        end_date = datetime(year, month_num + 1, 1, 0, 0, 0, 0) - timedelta(microseconds=1)
        except Exception as e:
            print(f"Ошибка парсинга 'месяц год': {e}")
            pass

    return start_date, end_date

def process_report_period(message):
    chat_id = message.chat.id
    user_id = message.from_user.id
    text = message.text.lower()
    
    family_id = get_user_active_family_id(user_id)
    if family_id is None: # Повторная проверка
        bot.send_message(chat_id, "Произошла ошибка: не удалось определить активную семью для отчета.")
        return
    if not is_family_subscription_active(family_id): # Повторная проверка
        bot.send_message(chat_id, "Ваша подписка истекла. Пожалуйста, продлите ее для получения отчетов.")
        return

    start_date, end_date = parse_date_period(text)

    if not start_date:
        bot.send_message(chat_id, "Не могу распознать период. Попробуйте 'сегодня', 'неделя', 'месяц', 'май 2025' или 'с 01.01.2024 по 31.01.2024'.")
        return

    conn = get_db_connection()
    if not conn:
        bot.send_message(chat_id, "Проблема с подключением к базе данных.")
        return
    
    cursor = conn.cursor()
    # Отчеты теперь берутся по family_id
    cursor.execute('''
        SELECT category, SUM(amount)
        FROM expenses
        WHERE family_id = %s AND timestamp BETWEEN %s AND %s
        GROUP BY category
    ''', (family_id, start_date, end_date))
    
    data = cursor.fetchall()
    conn.close()

    if not data:
        bot.send_message(chat_id, "За выбранный период нет расходов.")
        return

    categories = [row[0] for row in data]
    amounts = [row[1] for row in data]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(amounts, labels=categories, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
    ax.axis('equal')
    
    report_title_period = text.capitalize()
    if 'с ' in text_lower and ' по ' in text_lower and start_date and end_date:
        report_title_period = f"с {start_date.strftime('%d.%m.%Y')} по {end_date.strftime('%d.%m.%Y')}"
    elif 'сегодня' in text_lower:
        report_title_period = datetime.now().strftime('%d.%m.%Y')
    elif 'неделя' in text_lower:
        report_title_period = f"за неделю (до {datetime.now().strftime('%d.%m.%Y')})"
    elif 'месяц' in text_lower:
        report_title_period = f"за {datetime.now().strftime('%B %Y').lower()}"
    elif 'год' in text_lower:
        report_title_period = f"за {datetime.now().year} год"
    else:
        report_title_period = text_lower

    ax.set_title(f'Отчет о расходах за {report_title_period.capitalize()} (Тг)')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)

    bot.send_photo(chat_id, buf)

@bot.message_handler(func=lambda message: True)
def handle_expense_input(message):
    chat_id = message.chat.id
    user_id = message.from_user.id
    text = message.text.strip()

    family_id = get_user_active_family_id(user_id)
    if family_id is None:
        bot.send_message(chat_id, "Вы не состоите в активной семье. Для записи расходов вам нужно создать семью (админ) или присоединиться к существующей.")
        return
    
    # Проверка подписки перед записью расхода
    if not is_family_subscription_active(family_id):
        bot.send_message(chat_id, "Ваша подписка истекла. Пожалуйста, продлите ее для записи расходов.")
        return

    amount = None
    currency = 'тг'

    # Улучшенный парсинг: ищем сумму и валюту
    match_end = re.search(r'(\d+[\s\d.,]*)([а-яА-ЯёЁa-zA-Z$₽]{1,4})?$', text)
    if match_end:
        amount_str_raw = match_end.group(1).replace(' ', '').replace(',', '.')
        currency_part = match_end.group(2).strip().lower() if match_end.group(2) else ''
        try:
            amount = float(amount_str_raw)
            if currency_part in ['тг', 'kzt', 'тенге', '$', 'usd', 'руб', 'rub', '₽', 'eur']:
                currency = currency_part
            elif len(currency_part) <= 4 and currency_part.isalpha():
                currency = currency_part
            else: # Если валюта не распознана, но есть число, используем дефолт
                currency = 'тг' 
            description = text[:match_end.start()].strip()
        except ValueError:
            amount = None

    if amount is None:
        match_start = re.search(r'^(\d+[\s\d.,]*)(\s*[а-яА-ЯёЁa-zA-Z$₽]{1,4})?\s*(.*)', text)
        if match_start:
            amount_str_raw = match_start.group(1).replace(' ', '').replace(',', '.')
            currency_part = match_start.group(2).strip().lower() if match_start.group(2) else ''
            try:
                amount = float(amount_str_raw)
                if currency_part in ['тг', 'kzt', 'тенге', '$', 'usd', 'руб', 'rub', '₽', 'eur']:
                    currency = currency_part
                elif len(currency_part) <= 4 and currency_part.isalpha():
                    currency = currency_part
                else:
                    currency = 'тг'
                description = match_start.group(3).strip()
            except ValueError:
                amount = None
    
    if amount is None: # Попробуем формат "описание 1000"
        match_only_amount_end = re.search(r'(\d+[\s\d.,]*)$', text)
        if match_only_amount_end:
            amount_str_raw = match_only_amount_end.group(1).replace(' ', '').replace(',', '.')
            try:
                amount = float(amount_str_raw)
                description = text[:match_only_amount_end.start()].strip()
            except ValueError:
                amount = None
    
    if amount is None: # Попробуем "1000 описание"
        match_only_amount_start = re.search(r'^(\d+[\s\d.,]*)\s*(.*)', text)
        if match_only_amount_start:
            amount_str_raw = match_only_amount_start.group(1).replace(' ', '').replace(',', '.')
            try:
                amount = float(amount_str_raw)
                description = match_only_amount_start.group(2).strip()
            except ValueError:
                amount = None

    if amount is None:
        bot.send_message(chat_id, "Не могу найти сумму расхода. Пожалуйста, убедитесь, что сумма указана числом. Попробуйте 'описание сумма валюта', например: 'хлеб 100тг' или '1500тг бензин'.")
        return
    
    if not description:
        description = "Без описания" 

    # Классификация
    category = classify_expense(description)

    # Сохранение в БД
    conn = get_db_connection()
    if not conn:
        bot.send_message(chat_id, "Проблема с подключением к базе данных.")
        return
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO expenses (user_id, family_id, amount, currency, description, category)
            VALUES (%s, %s, %s, %s, %s, %s)
        ''', (user_id, family_id, amount, currency, description, category))
        conn.commit()
        bot.send_message(chat_id, f"Расход '{description} {amount}{currency}' добавлен в категорию '{category}'.")
    except Exception as e:
        conn.rollback()
        print(f"Ошибка при добавлении расхода в БД: {e}")
        bot.send_message(chat_id, "Произошла ошибка при сохранении расхода. Пожалуйста, попробуйте еще раз.")
    finally:
        if conn:
            conn.close()

# --- Фоновая задача для напоминаний ---
# Этот код будет выполняться в отдельном процессе/сервисе на Railway,
# который будет запускаться как "Worker" или "Cron Job".
# Он НЕ ДОЛЖЕН БЫТЬ в основном bot.py, если bot.py - это Telegram polling.
# Но для простоты примера, пока что оставим его здесь.
# В реальной системе это будет отдельный скрипт (например, reminder_worker.py)
# или функция, вызываемая Cron Job.
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
            SELECT id, user_id, family_id, title, next_due_date, reminder_offset_days, last_reminded_date
            FROM recurring_payments
            WHERE is_active = TRUE OR is_active IS NULL -- Если нет поля is_active или оно не используется
            """
        )
        reminders = cursor.fetchall()

        for r in reminders:
            rp_id, user_id, family_id, title, next_due_date, reminder_offset_days, last_reminded_date = r
            
            # Проверяем, пришло ли время напомнить
            days_until_due = (next_due_date - today).days
            
            if days_until_due == reminder_offset_days:
                # Проверим, напоминание уже отправлялось за этот период?
                # Если last_reminded_date не равно next_due_date - reminder_offset_days
                # или если оно вообще не было отправлено для текущего цикла
                if not last_reminded_date or last_reminded_date != today:
                    
                    message_text = (
                        f"🔔 Напоминание: **'{title}'**\n"
                        f"Следующий платеж: {next_due_date.strftime('%d.%m.%Y')}\n"
                        f"До него осталось {reminder_offset_days} дней."
                    )
                    
                    # Отправляем напоминание
                    try:
                        # Если семейный платеж, отправляем всем членам семьи, иначе лично пользователю
                        if family_id:
                            cursor.execute("SELECT user_id FROM user_families WHERE family_id = %s", (family_id,))
                            family_members = cursor.fetchall()
                            for member_id in family_members:
                                bot.send_message(member_id[0], message_text, parse_mode='Markdown')
                        else: # Если платеж личный (family_id IS NULL)
                            bot.send_message(user_id, message_text, parse_mode='Markdown')
                        
                        # Обновляем last_reminded_date
                        cursor.execute(
                            "UPDATE recurring_payments SET last_reminded_date = %s WHERE id = %s",
                            (today, rp_id)
                        )
                        conn.commit()
                        print(f"Отправлено напоминание для {title} (ID: {rp_id})")
                    except Exception as send_e:
                        print(f"Ошибка при отправке напоминания для {title} (ID: {rp_id}): {send_e}")
                        conn.rollback() # Откатываем, если не удалось отправить/обновить
                        
            # Логика для сдвига next_due_date на следующий период после того, как дата "прошла"
            # Это можно сделать, например, когда next_due_date <= today и last_reminded_date уже обновлена.
            # Для простоты пока оставим, что пользователь должен будет вручную обновить дату,
            # или это будет часть более сложной автоматизации.
            # Пример автоматического сдвига после истечения срока:
            if next_due_date <= today and last_reminded_date == today: # Проверка на то, что напоминание было отправлено в день платежа или уже после
                new_due_date = next_due_date
                if r[5] == 'YEAR':
                    new_due_date = new_due_date.replace(year=new_due_date.year + r[6])
                elif r[5] == 'MONTH':
                    new_due_date = new_due_date.replace(month=new_due_date.month + r[6])
                    if new_due_date.month > 12: # Обработка перехода года
                         new_due_date = new_due_date.replace(year=new_due_date.year + (new_due_date.month // 12), month=new_due_date.month % 12 or 12)
                elif r[5] == 'WEEK':
                    new_due_date = new_due_date + timedelta(weeks=r[6])
                elif r[5] == 'DAY':
                    new_due_date = new_due_date + timedelta(days=r[6])
                
                if new_due_date > next_due_date: # Только если дата действительно сдвинулась
                    cursor.execute(
                        "UPDATE recurring_payments SET next_due_date = %s, last_reminded_date = NULL WHERE id = %s",
                        (new_due_date, rp_id)
                    )
                    conn.commit()
                    print(f"Дата следующего платежа для '{title}' сдвинута на {new_due_date.strftime('%Y-%m-%d')}")

    except Exception as e:
        print(f"Критическая ошибка при проверке напоминаний: {e}")
    finally:
        conn.close()

# --- Запуск бота и фоновой задачи ---
if __name__ == '__main__':
    # Обучаем модель при запуске бота.
    train_model(TRAINING_DATA) 
    init_db() # Инициализируем БД при запуске (проверяем/создаем таблицы)
    
    print("Бот запущен...")
    # Запускаем фоновую задачу для напоминаний в отдельном потоке,
    # чтобы она не блокировала работу основного бота.
    # Это простой вариант для тестирования, но для продакшена лучше Cron Job или Worker.
    # import threading
    # reminder_thread = threading.Thread(target=run_reminder_worker)
    # reminder_thread.daemon = True # Позволяет потоку завершиться, если основная программа завершается
    # reminder_thread.start()

    # Основной цикл бота
    bot.polling(none_stop=True)

# Для отдельного Worker'а, который запускается ежедневно (например, через Cron Job на Railway)
# Вы можете создать отдельный файл, например, `reminder_worker.py`:
# --- Содержимое `reminder_worker.py` ---
# from bot import check_and_send_reminders
# if __name__ == '__main__':
#     check_and_send_reminders()
#     print("Ежедневная проверка напоминаний завершена.")
# ---
# На Railway в Cron Job вы бы указывали `python reminder_worker.py` и расписание `@daily`.
# В файле bot.py тогда бы убрали весь код check_and_send_reminders из __main__