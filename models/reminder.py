"""
Модель напоминания
"""
from dataclasses import dataclass
from datetime import datetime, date
from decimal import Decimal
from typing import Optional

@dataclass
class Reminder:
    """Модель напоминания"""
    id: Optional[int] = None
    user_id: int = 0
    title: str = ""
    description: Optional[str] = None
    amount: Decimal = Decimal('0')
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    is_active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        """Преобразует в словарь"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'title': self.title,
            'description': self.description,
            'amount': float(self.amount),
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Reminder':
        """Создает из словаря"""
        return cls(
            id=data.get('id'),
            user_id=data['user_id'],
            title=data['title'],
            description=data.get('description'),
            amount=Decimal(str(data['amount'])),
            start_date=datetime.fromisoformat(data['start_date']).date() if data.get('start_date') else None,
            end_date=datetime.fromisoformat(data['end_date']).date() if data.get('end_date') else None,
            is_active=data.get('is_active', True),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None,
            updated_at=datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else None
        )
