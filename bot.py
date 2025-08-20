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
log_directory = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_directory, exist_ok=True)
log_file_path = os.path.join(log_directory, 'finbot.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Загружаем переменные окружения
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

# --- TRAINING_DATA для ML модели ---
TRAINING_DATA = [
    ("каша", "Продукты"),
    ("гречка", "Продукты"),
    ("овсянка", "Продукты"),
    ("манка", "Продукты"),
    ("рис", "Продукты"),
    ("перловка", "Продукты"),
    ("пшено", "Продукты"),
    ("хлеб", "Продукты"),
    ("молоко", "Продукты"),
    ("мясо", "Продукты"),
    ("футболка", "Одежда"),
    ("джинсы", "Одежда"),
    ("ботинки", "Одежда"),
    ("телефон", "Электроника"),
    ("ноутбук", "Электроника"),
    ("бензин", "Авто"),
    ("лекарства", "Лекарства"),
    ("стиральный порошок", "Бытовая химия")
]

# --- Классификация расходов ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import re
import unicodedata

# Словарь категорий
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
        "каша","продукты","продукт","продуктовый","прод","еда","питание","бакалея","молочка","выпечка","овощи","фрукты"
    ],
    "Одежда": [
        "футболка","рубашка","кофта","свитер","толстовка","пиджак","жилет","пальто","куртка","плащ","шуба",
        "брюки","джинсы","шорты","юбка","платье","комбинезон","колготки","носки","гетры",
        "обувь","ботинки","туфли","кроссовки","кеды","сланцы","тапочки","сандалии",
        "одежда","шмот","шмотки","вещи","толстовки","толстовочка","кофточка"
    ],
    "Электроника": [
        "телефон","ноутбук","планшет","монитор","клавиатура","мышь","наушники","зарядка","пауэрбанк","телевизор","смарт-часы","колонка",
        "электроника","гаджеты","техника","кабель","адаптер","роутер","флешка","ssd","hdd"
    ],
    "Авто": [
        "бензин","дизель","масло моторное","антифриз","омыватель","шины","аккумулятор","тормозная жидкость","автомойка",
        "авто","машина","автомобиль","транспорт","колодки","фильтр"
    ],
    "Лекарства": [
        "парацетамол","ибупрофен","аспирин","но-шпа","пластырь","мазь","капли","витамины","анальгин","цитрамон",
        "лекарства","таблетки","аптека","лекарство","термометр","сироп","спрей"
    ],
    "Бытовая химия": [
        "стиральный порошок","кондиционер для белья","чистящее средство","средство для мытья посуды","отбеливатель",
        "доместос","фейри","санокс","антижир","химия","уборка","бытхимия","освежитель"
    ],
    "Прочее": [
        "подарок","сувенир","книга","журнал","газета","разное","прочее","непонятно","всякое","проч"
    ]
}

# Нормализация текста
def normalize(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    t = t.replace("ё","е")
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"[^a-zа-я0-9\s\-_/\.]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# Словарный матч
def dict_match_category(text_norm: str) -> str | None:
    for cat, words in CATEGORIES.items():
        for w in words:
            if w in text_norm:
                return cat
    return None

# Фуззи-матч
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

# ML модель
vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3,5),
    min_df=1,
    max_features=40000
)
classifier = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

def train_model(data):
    """Обучение модели"""
    use_data = data if (isinstance(data, list) and len(data) > 0) else TRAINING_DATA
    if not use_data:
        logger.warning("Нет данных для обучения модели.")
        return
    
    descriptions = [normalize(item[0]) for item in use_data]
    categories = [item[1] for item in use_data]
    X = vectorizer.fit_transform(descriptions)
    classifier.fit(X, categories)
    logger.info("Модель классификации успешно обучена.")

def classify_expense(description: str) -> str:
    """Классификация расхода"""
    try:
        text_norm = normalize(description)
        
        # 1) словарь
        cat = dict_match_category(text_norm)
        if cat:
            return cat
        
        # 2) фуззи
        cat = fuzzy_category(text_norm)
        if cat:
            return cat
        
        # 3) ML
        if hasattr(classifier, "classes_") and len(getattr(classifier, "classes_", [])) > 0:
            vec = vectorizer.transform([text_norm])
            pred = classifier.predict(vec)[0]
            return pred
        
        return "Прочее"
    except Exception as e:
        logger.error(f"Ошибка при классификации: {e}")
        return "Прочее"

# Обучаем модель
train_model(TRAINING_DATA)

# Остальной код бота...
def main():
    """Основная функция"""
    train_model(TRAINING_DATA)
    logger.info("FinBot запущен!")

if __name__ == "__main__":
    main() 