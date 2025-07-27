import os
import logging
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, filters
import psycopg2
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import io # Для работы с изображениями в памяти
from telegram.ext import ConversationHandler
import re

# --- Настройка логирования (для отладки) ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Получение переменных окружения из Railway ---
# Используем BOT_TOKEN, как это указано в ваших переменных Railway
BOT_TOKEN = os.getenv("BOT_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")
# ADMIN_USER_ID - если вы хотите использовать его для отправки уведомлений админу
ADMIN_USER_ID = os.getenv("ADMIN_USER_ID") 
if ADMIN_USER_ID:
    try:
        ADMIN_USER_ID = int(ADMIN_USER_ID)
    except ValueError:
        logger.warning("ADMIN_USER_ID is not a valid integer. It will be ignored.")
        ADMIN_USER_ID = None

# --- Функции для работы с базой данных ---

def get_db_connection():
    """Устанавливает соединение с базой данных PostgreSQL."""
    if not DATABASE_URL:
        logger.error("DATABASE_URL environment variable is not set.")
        raise ValueError("DATABASE_URL is not set. Cannot connect to database.")
    return psycopg2.connect(DATABASE_URL)

def create_table_if_not_exists():
    """
    Создаёт таблицу expenses, если она не существует, с нужными полями.
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS expenses (
                id SERIAL PRIMARY KEY,
                amount NUMERIC(10, 2) NOT NULL,
                description TEXT,
                category VARCHAR(100) NOT NULL,
                transaction_date TIMESTAMP WITH TIME ZONE NOT NULL
            );
        ''')
        conn.commit()
        logger.info("Таблица 'expenses' проверена/создана.")
    except Exception as e:
        logger.error(f"Ошибка при создании/обновлении таблицы: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

# --- Исправленная функция добавления расхода ---
def add_expense(amount, category, description, transaction_date):
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO expenses (amount, description, category, transaction_date)
            VALUES (%s, %s, %s, %s);
            """,
            (amount, description, category, transaction_date)
        )
        conn.commit()
        logger.info(f"Добавлена запись: {amount} {category} {description} {transaction_date}")
        return True
    except Exception as e:
        logger.error(f"Ошибка при добавлении расхода: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

# --- Исправленная функция выборки расходов для отчёта ---
def get_expenses_for_report(start_date: datetime, end_date: datetime):
    conn = None
    expenses = []
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT amount, category, description, transaction_date
            FROM expenses
            WHERE transaction_date BETWEEN %s AND %s
            ORDER BY transaction_date;
            """,
            (start_date, end_date)
        )
        expenses = cur.fetchall()
        logger.info(f"Получено {len(expenses)} записей для отчёта за период {start_date} - {end_date}")
    except Exception as e:
        logger.error(f"Ошибка при получении расходов для отчёта: {e}")
    finally:
        if conn:
            conn.close()
    return expenses

# --- Функции для Telegram бота ---

async def start(update: Update, context) -> None:
    """Отвечает на команду /start."""
    await update.message.reply_text(
        "Привет! Я бот для учета расходов. Отправь мне расход в формате: 'Сумма Категория Описание'\n"
        "Например: '150 Еда Обед в кафе'\n"
        "Для отчета используй команду /report."
    )

# --- Главное меню ---
def get_main_menu_keyboard():
    keyboard = [
        [KeyboardButton("Добавить расход"), KeyboardButton("📊 Отчеты")]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

# --- Клавиатура для выбора периода отчета ---
def get_report_period_keyboard():
    keyboard = [
        [KeyboardButton("Сегодня"), KeyboardButton("Неделя")],
        [KeyboardButton("Месяц"), KeyboardButton("Год")]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)

# --- Хэндлер для кнопки "📊 Отчеты" ---
async def menu(update: Update, context) -> None:
    await update.message.reply_text(
        "За какой период вы хотите отчет?",
        reply_markup=get_report_period_keyboard()
    )

# --- Хэндлер для выбора периода ---
PERIOD = 1
async def period_choice(update: Update, context) -> int:
    logger.info(f"Выбран период: {update.message.text}")
    text = update.message.text.lower()
    today = datetime.now(timezone.utc)
    if text in ["сегодня", "сегодня"]:
        start = today.replace(hour=0, minute=0, second=0, microsecond=0)
        end = today.replace(hour=23, minute=59, second=59, microsecond=999999)
    elif text in ["неделя", "неделя"]:
        start = (today - timedelta(days=6)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = today.replace(hour=23, minute=59, second=59, microsecond=999999)
    elif text in ["месяц", "месяц"]:
        start = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end = today.replace(hour=23, minute=59, second=59, microsecond=999999)
    elif text in ["год", "год"]:
        start = today.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        end = today.replace(hour=23, minute=59, second=59, microsecond=999999)
    else:
        await update.message.reply_text("Пожалуйста, выберите период с клавиатуры.", reply_markup=get_report_period_keyboard())
        return PERIOD
    await send_report(update, context, start, end)
    # После отчета возвращаем главное меню
    await update.message.reply_text("Главное меню:", reply_markup=get_main_menu_keyboard())
    return ConversationHandler.END

# --- Универсальная функция отправки отчета за период ---
async def send_report(update, context, start, end):
    try:
        expenses_data = get_expenses_for_report(start, end)
        if not expenses_data:
            await update.message.reply_text("За выбранный период расходы не найдены.")
            return
        total_amount = sum(float(e[0]) for e in expenses_data)
        report_text = f"📊 *Отчёт о расходах за период*\n\n"
        category_sums = {}
        for amount, category, _, _ in expenses_data:
            category_sums[category] = category_sums.get(category, 0) + float(amount)
        for category, amount in sorted(category_sums.items(), key=lambda item: item[1], reverse=True):
            report_text += f"*{category}:* {amount:.2f}\n"
        report_text += f"\n*Итого расходов: {total_amount:.2f}*"
        chart_buffer = generate_expense_chart(expenses_data, f"Расходы за период")
        if chart_buffer:
            await update.message.reply_photo(photo=chart_buffer, caption=report_text, parse_mode='Markdown')
        else:
            await update.message.reply_text(report_text, parse_mode='Markdown')
    except Exception as e:
        logger.error(f"Ошибка при формировании отчёта: {e}")
        await update.message.reply_text(f"Ошибка при формировании отчёта: {e}")

# --- ConversationHandler для меню отчетов ---
report_conv_handler = ConversationHandler(
    entry_points=[MessageHandler(filters.Regex("^📊 Отчеты$"), menu)],
    states={
        PERIOD: [MessageHandler(filters.Regex("^(Сегодня|Неделя|Месяц|Год)$", flags=re.IGNORECASE), period_choice)],
    },
    fallbacks=[],
    allow_reentry=True
)

async def handle_message(update: Update, context) -> None:
    """
    Обрабатывает текстовые сообщения для записи расходов.
    Теперь поддерживает формат: 'Описание сумма' (например, 'Хлеб 130').
    Игнорирует сообщения-кнопки главного меню и меню отчетов.
    """
    text = update.message.text.strip()
    # Не реагируем на кнопки меню и периода
    if text in ["Добавить расход", "📊 Отчеты", "Сегодня", "Неделя", "Месяц", "Год"]:
        return
    logger.info(f"Получено сообщение от {update.message.from_user.id}: {text}")
    import re
    match = re.match(r"(.+?)\s+(\d+[.,]?\d*)$", text)
    if not match:
        await update.message.reply_text(
            "Неверный формат. Используйте: 'Описание сумма' (например, 'Хлеб 130').",
            reply_markup=get_main_menu_keyboard()
        )
        return
    description = match.group(1).strip()
    amount_str = match.group(2).replace(',', '.')
    try:
        amount = float(amount_str)
        category = "Прочее"
        transaction_date = datetime.now(timezone.utc)
        if add_expense(amount, category, description, transaction_date):
            await update.message.reply_text(
                f"Расход {amount:.2f} ({description}) записан!",
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

# --- Функции для генерации отчетов и диаграмм ---

def generate_expense_chart(expenses_data, title="Расходы по категориям"):
    category_sums = {}
    for amount, category, _, _ in expenses_data:
        category_sums[category] = category_sums.get(category, 0) + float(amount)
    if not category_sums:
        return None
    labels = list(category_sums.keys())
    sizes = list(category_sums.values())
    colors = plt.cm.Paired.colors
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors,
            wedgeprops={'edgecolor': 'black', 'linewidth': 0.5, 'antialiased': True},
            textprops={'fontsize': 10, 'color': 'black', 'weight': 'bold'})
    ax1.axis('equal')
    plt.title(title, fontsize=16, pad=20, weight='bold')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close(fig1)
    return buf

async def report(update: Update, context) -> None:
    logger.info(f"Получена команда /report от {update.message.from_user.id}")
    await update.message.reply_text("Формирую отчёт за текущий месяц...")
    today = datetime.now(timezone.utc)
    start_of_month = datetime(today.year, today.month, 1, 0, 0, 0, tzinfo=timezone.utc)
    end_of_day = datetime(today.year, today.month, today.day, 23, 59, 59, tzinfo=timezone.utc)
    try:
        expenses_data = get_expenses_for_report(start_of_month, end_of_day)
        if not expenses_data:
            await update.message.reply_text("За текущий месяц расходы не найдены.")
            return
        total_amount = sum(float(e[0]) for e in expenses_data)
        report_text = f"📊 *Отчёт о расходах за {start_of_month.strftime('%B %Y')}*\n\n"
        category_sums = {}
        for amount, category, _, _ in expenses_data:
            category_sums[category] = category_sums.get(category, 0) + float(amount)
        for category, amount in sorted(category_sums.items(), key=lambda item: item[1], reverse=True):
            report_text += f"*{category}:* {amount:.2f}\n"
        report_text += f"\n*Итого расходов: {total_amount:.2f}*"
        chart_buffer = generate_expense_chart(expenses_data, f"Расходы за {start_of_month.strftime('%B %Y')}")
        if chart_buffer:
            await update.message.reply_photo(photo=chart_buffer, caption=report_text, parse_mode='Markdown')
        else:
            await update.message.reply_text(report_text, parse_mode='Markdown')
    except Exception as e:
        logger.error(f"Ошибка при формировании отчёта: {e}")
        await update.message.reply_text(f"Ошибка при формировании отчёта: {e}")

# --- Главная функция запуска бота ---

def main():
    """Запускает бота."""
    # Убедимся, что токен бота установлен
    if not BOT_TOKEN:
        logger.error("BOT_TOKEN environment variable is not set. Bot cannot start.")
        raise ValueError("BOT_TOKEN is not set.")

    # Убедимся, что таблица создана/обновлена перед запуском бота
    # create_table_if_not_exists() вызывается здесь для инициализации БД
    # Обратите внимание: функция create_table_if_not_exists() должна быть потокобезопасной
    # или вызываться до инициализации Application.
    # В текущей структуре она вызывается синхронно до run_polling, что нормально.
    create_table_if_not_exists() 

    application = Application.builder().token(BOT_TOKEN).build() # Использование BOT_TOKEN

    # Обработчики команд и сообщений
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("report", report))
    application.add_handler(report_conv_handler)  # ConversationHandler должен идти ДО обычного MessageHandler
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Бот запущен!")
    application.run_polling()

if __name__ == "__main__":
    main()