"""
Валидаторы для входных данных
"""
import re
from datetime import datetime, date
from typing import Union, Optional
from decimal import Decimal, InvalidOperation
from utils.exceptions import ValidationError

class Validator:
    """Базовый класс валидатора"""
    
    @staticmethod
    def validate_not_empty(value: str, field_name: str = "Поле") -> str:
        """Проверка на пустое значение"""
        if not value or not value.strip():
            raise ValidationError(f"{field_name} не может быть пустым")
        return value.strip()
    
    @staticmethod
    def validate_length(value: str, min_length: int = 1, max_length: int = 255, field_name: str = "Поле") -> str:
        """Проверка длины строки"""
        value = Validator.validate_not_empty(value, field_name)
        if len(value) < min_length:
            raise ValidationError(f"{field_name} должно содержать минимум {min_length} символов")
        if len(value) > max_length:
            raise ValidationError(f"{field_name} должно содержать максимум {max_length} символов")
        return value
    
    @staticmethod
    def validate_amount(amount_str: str) -> Decimal:
        """Валидация суммы"""
        if not amount_str:
            raise ValidationError("Сумма не может быть пустой")
        
        # Заменяем запятую на точку
        amount_str = amount_str.replace(',', '.')
        
        try:
            amount = Decimal(amount_str)
            if amount <= 0:
                raise ValidationError("Сумма должна быть больше нуля")
            if amount > Decimal('999999999.99'):
                raise ValidationError("Сумма слишком большая")
            return amount
        except InvalidOperation:
            raise ValidationError("Неверный формат суммы")
    
    @staticmethod
    def validate_date(date_str: str, format_str: str = "%d.%m.%Y") -> date:
        """Валидация даты"""
        if not date_str:
            raise ValidationError("Дата не может быть пустой")
        
        try:
            return datetime.strptime(date_str.strip(), format_str).date()
        except ValueError:
            raise ValidationError(f"Неверный формат даты. Используйте формат: {format_str}")
    
    @staticmethod
    def validate_month_year(month_year_str: str) -> tuple[int, int]:
        """Валидация месяца и года (ММ.ГГГГ)"""
        if not month_year_str:
            raise ValidationError("Месяц и год не могут быть пустыми")
        
        if not re.match(r'^\d{1,2}\.\d{4}$', month_year_str.strip()):
            raise ValidationError("Неверный формат. Используйте ММ.ГГГГ (например: 09.2025)")
        
        try:
            month, year = month_year_str.strip().split('.')
            month = int(month)
            year = int(year)
            
            if not (1 <= month <= 12):
                raise ValidationError("Месяц должен быть от 1 до 12")
            
            if not (2020 <= year <= 2030):
                raise ValidationError("Год должен быть от 2020 до 2030")
            
            return month, year
        except ValueError:
            raise ValidationError("Неверный формат месяца и года")
    
    @staticmethod
    def validate_username(username: str) -> str:
        """Валидация имени пользователя"""
        username = Validator.validate_length(username, 2, 50, "Имя пользователя")
        
        # Проверяем, что содержит только буквы, цифры, пробелы и дефисы
        if not re.match(r'^[a-zA-Zа-яА-Я0-9\s\-]+$', username):
            raise ValidationError("Имя пользователя может содержать только буквы, цифры, пробелы и дефисы")
        
        return username
    
    @staticmethod
    def validate_category(category: str) -> str:
        """Валидация категории"""
        return Validator.validate_length(category, 1, 100, "Категория")
    
    @staticmethod
    def validate_description(description: str) -> str:
        """Валидация описания"""
        return Validator.validate_length(description, 1, 500, "Описание")
    
    @staticmethod
    def validate_phone(phone: str) -> str:
        """Валидация номера телефона"""
        if not phone:
            raise ValidationError("Номер телефона не может быть пустым")
        
        # Убираем все символы кроме цифр и +
        clean_phone = re.sub(r'[^\d+]', '', phone)
        
        # Проверяем формат
        if not re.match(r'^\+?[1-9]\d{6,14}$', clean_phone):
            raise ValidationError("Неверный формат номера телефона")
        
        return clean_phone
    
    @staticmethod
    def validate_invitation_code(code: str) -> str:
        """Валидация кода приглашения"""
        if not code:
            raise ValidationError("Код приглашения не может быть пустым")
        
        if not re.match(r'^[A-Z0-9]{6,12}$', code.upper()):
            raise ValidationError("Код приглашения должен содержать 6-12 символов (буквы и цифры)")
        
        return code.upper()
    
    @staticmethod
    def validate_choice(choice_str: str, max_choice: int, field_name: str = "Выбор") -> int:
        """Валидация выбора из списка"""
        if not choice_str:
            raise ValidationError(f"{field_name} не может быть пустым")
        
        try:
            choice = int(choice_str.strip())
            if not (1 <= choice <= max_choice):
                raise ValidationError(f"{field_name} должен быть от 1 до {max_choice}")
            return choice
        except ValueError:
            raise ValidationError(f"Неверный формат {field_name.lower()}")
