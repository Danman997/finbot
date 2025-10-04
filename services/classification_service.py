"""
Сервис для классификации расходов
"""
import re
import unicodedata
from typing import Dict, List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os
from utils import logger, DatabaseError

class ClassificationService:
    """Сервис для классификации расходов"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.categories = {}
        self.is_trained = False
        self.model_path = "models/classification_model.pkl"
        self.vectorizer_path = "models/vectorizer.pkl"
    
    def _normalize_text(self, text: str) -> str:
        """Нормализация текста для классификации"""
        if not text:
            return ""
        
        # Приводим к нижнему регистру
        text = text.lower()
        
        # Удаляем знаки препинания и цифры
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Удаляем лишние пробелы
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _load_categories(self):
        """Загрузка категорий из файла или базы данных"""
        # Базовые категории
        self.categories = {
            "Продукты": [
                "хлеб", "батон", "булочка", "молоко", "кефир", "мясо", "рыба", "овощи", "фрукты",
                "продукты", "еда", "питание", "бакалея", "молочка", "выпечка"
            ],
            "Транспорт": [
                "бензин", "топливо", "такси", "автобус", "метро", "поезд", "самолет",
                "транспорт", "проезд", "парковка", "штраф"
            ],
            "Развлечения": [
                "кино", "театр", "концерт", "ресторан", "кафе", "бар", "клуб",
                "развлечения", "отдых", "игры", "хобби"
            ],
            "Здоровье": [
                "врач", "больница", "лекарства", "аптека", "стоматолог",
                "здоровье", "медицина", "лечение"
            ],
            "Одежда": [
                "футболка", "рубашка", "брюки", "платье", "обувь", "куртка",
                "одежда", "магазин", "покупка"
            ],
            "Коммунальные услуги": [
                "электричество", "газ", "вода", "отопление", "интернет", "телефон",
                "коммунальные", "услуги", "счет"
            ],
            "Образование": [
                "школа", "университет", "курсы", "книги", "учебники",
                "образование", "обучение", "студент"
            ],
            "Прочее": [
                "подарок", "пожертвование", "страхование", "ремонт",
                "прочее", "другое", "разное"
            ]
        }
    
    def _prepare_training_data(self) -> Tuple[List[str], List[str]]:
        """Подготовка обучающих данных"""
        texts = []
        labels = []
        
        for category, keywords in self.categories.items():
            for keyword in keywords:
                texts.append(self._normalize_text(keyword))
                labels.append(category)
        
        return texts, labels
    
    def train_model(self, additional_data: List[Tuple[str, str]] = None):
        """Обучение модели классификации"""
        try:
            # Загружаем категории
            self._load_categories()
            
            # Подготавливаем базовые данные
            texts, labels = self._prepare_training_data()
            
            # Добавляем дополнительные данные если есть
            if additional_data:
                for text, label in additional_data:
                    texts.append(self._normalize_text(text))
                    labels.append(label)
            
            if not texts:
                logger.warning("Нет данных для обучения модели")
                return
            
            # Обучаем векторизатор
            self.vectorizer.fit(texts)
            
            # Преобразуем тексты в векторы
            X = self.vectorizer.transform(texts)
            
            # Обучаем модель
            self.model.fit(X, labels)
            
            self.is_trained = True
            
            # Сохраняем модель
            self._save_model()
            
            logger.info(f"Модель классификации обучена на {len(texts)} примерах")
            
        except Exception as e:
            logger.error(f"Ошибка обучения модели: {e}")
            raise DatabaseError(f"Не удалось обучить модель: {e}")
    
    def _save_model(self):
        """Сохранение модели и векторизатора"""
        try:
            os.makedirs("models", exist_ok=True)
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            with open(self.vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            logger.info("Модель и векторизатор сохранены")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения модели: {e}")
    
    def _load_model(self):
        """Загрузка модели и векторизатора"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                
                with open(self.vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                
                self.is_trained = True
                logger.info("Модель и векторизатор загружены")
                
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            self.is_trained = False
    
    def classify_expense(self, description: str) -> str:
        """Классификация расхода по описанию"""
        try:
            if not self.is_trained:
                self._load_model()
            
            if not self.is_trained:
                # Если модель не обучена, используем словарный подход
                return self._classify_by_dictionary(description)
            
            # Нормализуем текст
            normalized_text = self._normalize_text(description)
            
            if not normalized_text:
                return "Прочее"
            
            # Преобразуем в вектор
            X = self.vectorizer.transform([normalized_text])
            
            # Получаем предсказание
            prediction = self.model.predict(X)[0]
            
            # Получаем вероятности
            probabilities = self.model.predict_proba(X)[0]
            max_probability = max(probabilities)
            
            # Если вероятность низкая, используем словарный подход
            if max_probability < 0.3:
                return self._classify_by_dictionary(description)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Ошибка классификации: {e}")
            return self._classify_by_dictionary(description)
    
    def _classify_by_dictionary(self, description: str) -> str:
        """Классификация по словарю"""
        if not description:
            return "Прочее"
        
        description_lower = description.lower()
        
        # Загружаем категории если не загружены
        if not self.categories:
            self._load_categories()
        
        # Ищем совпадения
        for category, keywords in self.categories.items():
            for keyword in keywords:
                if keyword in description_lower:
                    return category
        
        return "Прочее"
    
    def get_classification_confidence(self, description: str) -> float:
        """Получение уверенности в классификации"""
        try:
            if not self.is_trained:
                return 0.5  # Средняя уверенность для словарного подхода
            
            normalized_text = self._normalize_text(description)
            if not normalized_text:
                return 0.0
            
            X = self.vectorizer.transform([normalized_text])
            probabilities = self.model.predict_proba(X)[0]
            
            return float(max(probabilities))
            
        except Exception as e:
            logger.error(f"Ошибка получения уверенности: {e}")
            return 0.5
    
    def retrain_with_feedback(self, description: str, correct_category: str):
        """Переобучение модели с обратной связью"""
        try:
            # Добавляем новый пример
            normalized_text = self._normalize_text(description)
            
            if not normalized_text:
                return
            
            # Получаем текущие данные
            texts, labels = self._prepare_training_data()
            
            # Добавляем новый пример
            texts.append(normalized_text)
            labels.append(correct_category)
            
            # Переобучаем модель
            self.vectorizer.fit(texts)
            X = self.vectorizer.transform(texts)
            self.model.fit(X, labels)
            
            # Сохраняем модель
            self._save_model()
            
            logger.info(f"Модель переобучена с новым примером: {description} -> {correct_category}")
            
        except Exception as e:
            logger.error(f"Ошибка переобучения модели: {e}")

# Глобальный экземпляр сервиса
classification_service = ClassificationService()
