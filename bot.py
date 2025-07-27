import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
import psycopg2
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import io # Для работы с изображениями в памяти

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
    """Создает таблицу expenses и триггер, если они не существуют."""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Создание таблицы expenses
        cur.execute('''
            CREATE TABLE IF NOT EXISTS expenses (
                id SERIAL PRIMARY KEY,
                amount NUMERIC(10, 2) NOT NULL,
                type VARCHAR(50) NOT NULL DEFAULT 'expense',
                category VARCHAR(100) NOT NULL,
                description TEXT,
                transaction_date TIMESTAMP WITH TIME ZONE NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
        ''')
        
        # Создание функции-триггера для updated_at (если не существует или требует обновления)
        cur.execute('''
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = NOW();
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
        ''')
        
        # Создание триггера (удаляем, если уже есть, и создаем заново, чтобы быть уверенными)
        cur.execute('''
            DROP TRIGGER IF EXISTS update_expenses_updated_at ON expenses;
            CREATE TRIGGER update_expenses_updated_at
            BEFORE UPDATE ON expenses
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
        ''')
        conn.commit()
        logger.info("Таблица 'expenses' и триггер проверены/созданы.")
    except Exception as e:
        logger.error(f"Ошибка при создании/обновлении таблицы: {e}")
        if conn:
            conn.rollback()
        # Возможно, отправить сообщение админу, если ADMIN_USER_ID установлен
        # if ADMIN_USER_ID:
        #     async def send_error_to_admin():
        #         await application.bot.send_message(chat_id=ADMIN_USER_ID, text=f"Ошибка БД: {e}")
        #     # Это потребует настройки, чтобы application был доступен здесь.
        #     # Для простоты, пока просто логируем.
    finally:
        if conn:
            conn.close()

def add_expense(amount, category, description, transaction_date, expense_type='expense'):
    """Добавляет новую запись о расходе/доходе в базу данных."""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO expenses (amount, type, category, description, transaction_date)
            VALUES (%s, %s, %s, %s, %s);
            """,
            (amount, expense_type, category, description, transaction_date)
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

def get_expenses_for_report(start_date: datetime, end_date: datetime, expense_type='expense'):
    """Получает расходы за указанный период."""
    conn = None
    expenses = []
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT amount, category, description, transaction_date
            FROM expenses
            WHERE transaction_date BETWEEN %s AND %s AND type = %s
            ORDER BY transaction_date;
            """,
            (start_date, end_date, expense_type)
        )
        expenses = cur.fetchall()
        logger.info(f"Получено {len(expenses)} записей для отчета за период {start_date} - {end_date}")
    except Exception as e:
        logger.error(f"Ошибка при получении расходов для отчета: {e}")
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

async def handle_message(update: Update, context) -> None:
    """Обрабатывает текстовые сообщения для записи расходов."""
    text = update.message.text
    logger.info(f"Получено сообщение от {update.message.from_user.id}: {text}")

    try:
        parts = text.split(maxsplit=2) # Сумма, Категория, Описание

        if len(parts) < 2:
            await update.message.reply_text(
                "Неверный формат. Используйте: 'Сумма Категория Описание' (Описание необязательно)."
            )
            return

        amount = float(parts[0])
        category = parts[1]
        description = parts[2] if len(parts) > 2 else ""
        transaction_date = datetime.now(timezone.utc) # UTC время для consistent storage

        if add_expense(amount, category, description, transaction_date):
            await update.message.reply_text(
                f"Расход {amount:.2f} ({category}) записан!" # Форматируем сумму
            )
        else:
            await update.message.reply_text(
                "Произошла ошибка при записи расхода. Пожалуйста, попробуйте еще раз."
            )

    except ValueError:
        await update.message.reply_text(
            "Неверный формат суммы. Сумма должна быть числом (например, 150.50)."
        )
    except Exception as e:
        logger.error(f"Непредвиденная ошибка при обработке сообщения: {e}")
        await update.message.reply_text(f"Произошла непредвиденная ошибка: {e}")

# --- Функции для генерации отчетов и диаграмм ---

def generate_expense_chart(expenses_data, title="Расходы по категориям"):
    """
    Генерирует круговую диаграмму расходов по категориям.
    Возвращает BytesIO объект с изображением.
    """
    category_sums = {}
    for amount, category, _, _ in expenses_data:
        category_sums[category] = category_sums.get(category, 0) + float(amount)

    if not category_sums:
        return None

    labels = list(category_sums.keys())
    sizes = list(category_sums.values())
    
    # Цветовая палитра для диаграммы
    colors = plt.cm.Paired.colors # Можно выбрать другую палитру: tab10, Dark2, Set3, etc.
    
    # Создаем диаграмму
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors,
            wedgeprops={'edgecolor': 'black', 'linewidth': 0.5, 'antialiased': True},
            textprops={'fontsize': 10, 'color': 'black', 'weight': 'bold'})
    
    ax1.axis('equal')  # Гарантирует, что круговая диаграмма будет круглой.
    plt.title(title, fontsize=16, pad=20, weight='bold') # Заголовок диаграммы
    plt.tight_layout() # Автоматически корректирует параметры подложки для плотного расположения

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150) # Сохраняем в буфер
    buf.seek(0) # Перемещаем указатель в начало буфера
    plt.close(fig1) # Закрываем фигуру, чтобы освободить память
    return buf

async def report(update: Update, context) -> None:
    """Генерирует и отправляет отчет о расходах за месяц."""
    logger.info(f"Получена команда /report от {update.message.from_user.id}")
    
    await update.message.reply_text("Формирую отчет за текущий месяц...")

    today = datetime.now(timezone.utc)
    # Отчет за текущий календарный месяц (с 1-го числа до текущей даты)
    start_of_month = datetime(today.year, today.month, 1, 0, 0, 0, tzinfo=timezone.utc)
    # Конец текущего дня
    end_of_day = datetime(today.year, today.month, today.day, 23, 59, 59, tzinfo=timezone.utc)

    expenses_data = get_expenses_for_report(start_of_month, end_of_day, 'expense')

    if not expenses_data:
        await update.message.reply_text("За текущий месяц расходы не найдены.")
        return

    # Подготавливаем текстовый отчет
    total_amount = sum(float(e[0]) for e in expenses_data)
    report_text = f"📊 *Отчет о расходах за {start_of_month.strftime('%B %Y')}*\n\n"
    
    category_sums = {}
    for amount, category, _, _ in expenses_data:
        category_sums[category] = category_sums.get(category, 0) + float(amount)

    for category, amount in sorted(category_sums.items(), key=lambda item: item[1], reverse=True):
        report_text += f"*{category}:* {amount:.2f}\n"
    
    report_text += f"\n*Итого расходов: {total_amount:.2f}*"

    # Генерируем диаграмму
    chart_buffer = generate_expense_chart(expenses_data, f"Расходы за {start_of_month.strftime('%B %Y')}")

    if chart_buffer:
        await update.message.reply_photo(photo=chart_buffer, caption=report_text, parse_mode='Markdown')
    else:
        await update.message.reply_text(report_text, parse_mode='Markdown')


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
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Бот запущен!")
    application.run_polling()

if __name__ == "__main__":
    main()