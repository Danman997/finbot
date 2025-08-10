import os
import io
import re
import time
import logging
import unicodedata
from datetime import datetime, timedelta, timezone

import psycopg2
from psycopg2 import sql
from psycopg2 import errors as pg_errors

from dotenv import load_dotenv

from telegram import (
    Update,
    ReplyKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardRemove,
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    ContextTypes,
    filters,
)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import schedule

# =========================
# ЛОГИРОВАНИЕ
# =========================
log_directory = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_directory, exist_ok=True)
log_file_path = os.path.join(log_directory, 'finbot.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =========================
# ОКРУЖЕНИЕ
# =========================
load_dotenv()

BOT_TOKEN = os.getenv('BOT_TOKEN')
if not BOT_TOKEN:
    logger.error("Ошибка: Токен бота не найден. Установите переменную окружения BOT_TOKEN.")
    raise SystemExit(1)

DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    logger.error("Ошибка: URL базы данных не найден. Установите переменную окружения DATABASE_URL.")
    raise SystemExit(1)

# =========================
# КЛАССИФИКАЦИЯ (СЛОВАРЬ → ФУЗЗИ → ML)
# =========================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Базовый словарь категорий с синонимами/однокоренными
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
    "Прочее": [
        "подарок","сувенир","книга","журнал","газета","разное","прочее","непонятно","всякое","проч"
    ]
}

def normalize(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    t = t.replace("ё", "е")
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"[^a-zа-я0-9\s\-_/\.]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def dict_match_category(text_norm: str) -> str | None:
    for cat, words in CATEGORIES.items():
        for w in words:
            if w in text_norm:
                return cat
    return None

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

vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=1,
    max_features=40000
)
classifier = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

# Для совместимости со старым вызовом train_model(TRAINING_DATA)
TRAINING_DATA = []

BASE_TRAIN = []
for cat, words in CATEGORIES.items():
    for w in words:
        BASE_TRAIN.append((w, cat))

def train_model(data):
    use_data = data if (isinstance(data, list) and len(data) > 0) else BASE_TRAIN
    if not use_data:
        logger.warning("Нет данных для обучения модели. Модель не будет обучена.")
        return
    descriptions = [normalize(item[0]) for item in use_data]
    categories = [item[1] for item in use_data]
    X = vectorizer.fit_transform(descriptions)
    classifier.fit(X, categories)
    logger.info("Модель классификации (гибрид) успешно обучена.")

train_model(BASE_TRAIN)

def get_db_connection():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        logger.error(f"Ошибка подключения к БД: {e}")
        return None

def get_override_category(description: str) -> str | None:
    """Проверяем таблицу category_overrides: если для нормализованного описания есть ручное правило — используем его."""
    try:
        conn = get_db_connection()
        if not conn:
            return None
        cur = conn.cursor()
        norm = normalize(description)
        cur.execute("SELECT category FROM category_overrides WHERE norm_desc = %s LIMIT 1;", (norm,))
        row = cur.fetchone()
        conn.close()
        return row[0] if row else None
    except Exception as e:
        logger.error(f"Ошибка чтения override из БД: {e}")
        return None

def classify_expense(description: str) -> str:
    """
    Порядок: 0) ручной override в БД → 1) словарь → 2) фуззи → 3) ML → 4) 'Прочее'
    """
    try:
        # 0) ручное переопределение (обучение на правках)
        cat = get_override_category(description)
        if cat:
            logger.info(f"[override] '{description}' -> {cat}")
            return cat

        text_norm = normalize(description)

        # 1) словарь
        cat = dict_match_category(text_norm)
        if cat:
            logger.info(f"[dict] '{description}' -> {cat}")
            return cat

        # 2) фуззи
        cat = fuzzy_category(text_norm)
        if cat:
            logger.info(f"[fuzzy] '{description}' -> {cat}")
            return cat

        # 3) ML
        if hasattr(classifier, "classes_") and len(getattr(classifier, "classes_", [])) > 0:
            vec = vectorizer.transform([text_norm])
            pred = classifier.predict(vec)[0]
            logger.info(f"[ml] '{description}' -> {pred}")
            return pred

        # 4) fallback
        logger.info(f"[fallback] '{description}' -> Прочее")
        return "Прочее"
    except Exception as e:
        logger.error(f"Ошибка при классификации: {e}. Возвращаю 'Прочее'.")
        return "Прочее"

# =========================
# БАЗА ДАННЫХ (DDL/CRUD)
# =========================
def init_db():
    conn = get_db_connection()
    if not conn:
        return
    cursor = conn.cursor()
    try:
        # Чистим старые столбцы, если есть
        for col in ['user_id', 'family_id']:
            try:
                cursor.execute(f"ALTER TABLE expenses DROP COLUMN {col};")
                conn.commit()
                logger.info(f"Столбец {col} успешно удален из таблицы expenses.")
            except pg_errors.UndefinedColumn:
                conn.rollback()
            except Exception as e:
                conn.rollback()
                logger.error(f"Ошибка при удалении столбца {col}: {e}")

        # Основная таблица расходов
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS expenses (
                id SERIAL PRIMARY KEY,
                amount NUMERIC(10, 2) NOT NULL,
                description TEXT,
                category VARCHAR(100) NOT NULL,
                transaction_date TIMESTAMPTZ NOT NULL,
                predicted_category VARCHAR(100)
            );
        ''')
        conn.commit()

        # Таблица переопределений
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS category_overrides (
                id SERIAL PRIMARY KEY,
                norm_desc TEXT NOT NULL UNIQUE,
                category  VARCHAR(100) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_category_overrides_norm ON category_overrides (norm_desc);')
        conn.commit()

        # Функция-триггер: учимся на ручных правках категорий
        cursor.execute("""
        CREATE OR REPLACE FUNCTION learn_override_from_edit() RETURNS trigger AS $$
        DECLARE
          norm TEXT;
        BEGIN
          IF TG_OP = 'UPDATE' AND NEW.category IS DISTINCT FROM OLD.category THEN
            norm := lower(
                      regexp_replace(
                        translate(NEW.description, 'Ёё', 'Ее'),
                        '[^a-zA-Zа-яА-Я0-9\\s\\-\\_\\/\\.]', ' ', 'g'
                      )
                    );
            norm := regexp_replace(norm, '\\s+', ' ', 'g');

            INSERT INTO category_overrides(norm_desc, category)
            VALUES (norm, NEW.category)
            ON CONFLICT (norm_desc)
              DO UPDATE SET category = EXCLUDED.category, created_at = now();
          END IF;
          RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """)
        conn.commit()

        cursor.execute("DROP TRIGGER IF EXISTS trg_learn_override ON expenses;")
        cursor.execute("""
        CREATE TRIGGER trg_learn_override
        AFTER UPDATE OF category ON expenses
        FOR EACH ROW
        EXECUTE FUNCTION learn_override_from_edit();
        """)
        conn.commit()

        logger.info("База данных инициализирована (expenses/category_overrides/trigger созданы/проверены).")
    finally:
        conn.close()

def add_expense(amount, category, description, transaction_date):
    conn = get_db_connection()
    if not conn:
        return False
    try:
        cursor = conn.cursor()
        # сохраняем и предсказание модели для аудита
        try:
            text_norm = normalize(description)
            if hasattr(classifier, 'classes_') and len(classifier.classes_) > 0:
                pred = classifier.predict(vectorizer.transform([text_norm]))[0]
            else:
                pred = category
        except Exception:
            pred = category

        cursor.execute('''
            INSERT INTO expenses (amount, category, description, transaction_date, predicted_category)
            VALUES (%s, %s, %s, %s, %s)
        ''', (amount, category, description, transaction_date, pred))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Ошибка при добавлении расхода: {e}")
        return False
    finally:
        conn.close()

# =========================
# UI КЛАВИАТУРЫ
# =========================
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

# =========================
# КОМАНДЫ/ХЕНДЛЕРЫ
# =========================
PERIOD_CHOICE_STATE = 1

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Я твой помощник по учету расходов. Выбери опцию ниже:",
        reply_markup=get_main_menu_keyboard()
    )

async def report_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "За какой период вы хотите отчет?",
        reply_markup=get_report_period_keyboard()
    )
    return PERIOD_CHOICE_STATE

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

    # Создание DataFrame с полными данными (ОТЧЁТЫ — НЕ МЕНЯЕМ)
    try:
        df = pd.DataFrame(data, columns=['Описание', 'Категория', 'Сумма', 'Дата транзакции'])
        grouped_data = df.groupby('Категория', as_index=False)['Сумма'].sum().sort_values(by='Сумма', ascending=False)
        categories = grouped_data['Категория'].tolist()
        amounts = grouped_data['Сумма'].tolist()
        total = df['Сумма'].sum()
    except Exception as e:
        logger.error(f"Ошибка при создании DataFrame: {e}")
        return ConversationHandler.END

    # Excel
    excel_buf = io.BytesIO()
    df.to_excel(excel_buf, index=False, engine='xlsxwriter')
    excel_buf.seek(0)

    # График (ОТЧЁТЫ — НЕ МЕНЯЕМ)
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(amounts, labels=None, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
    ax.axis('equal')
    plt.title(f'Отчет о расходах за {period_text.capitalize()} (Тг)')

    legend_labels = [f"{cat} — {amt:.2f} Тг" for cat, amt in zip(categories, amounts)]
    plt.legend(wedges, legend_labels, title="Категории", loc="lower center", bbox_to_anchor=(0.5, -0.15), fontsize=12)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close(fig)

    table_text = "\n".join([f"{cat}: {amt:.2f} Тг" for cat, amt in zip(categories, amounts)])
    table_text += f"\n\nИтого: {total:.2f} Тг"

    await update.message.reply_photo(photo=buf, caption=table_text, reply_markup=get_main_menu_keyboard())
    await update.message.reply_document(document=excel_buf, filename=f"Отчет_{period_text}.xlsx")
    return ConversationHandler.END

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text_msg = update.message.text.strip()
    if text_msg in ["💸 Добавить расход", "📊 Отчеты", "Сегодня", "Неделя", "Месяц", "Год"]:
        return

    logger.info(f"Получено сообщение: {text_msg}")

    # Ожидаем формат "Описание Сумма"
    match = re.match(r"(.+?)\s+(\d+[.,]?\d*)$", text_msg)
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

# =========================
# MAIN + ежедневное дообучение
# =========================
def main():
    train_model(TRAINING_DATA)  # совместимость — обучит на BASE_TRAIN при пустом списке
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

    # === Ежедневное обучение (останется, как у вас было; сработает после остановки polling) ===
    def daily_training():
        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute('SELECT description, category FROM expenses')
                data = cursor.fetchall()
                conn.close()

                if data:
                    logger.info(f"Данные для обучения модели: {len(data)} записей")
                    descriptions = [normalize(row[0]) for row in data]
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

    schedule.every().day.at("00:00").do(daily_training)

    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()
