"""
Модель расхода
"""
from dataclasses import dataclass
from datetime import datetime, date
from decimal import Decimal
from typing import Optional

@dataclass
class Expense:
    """Модель расхода"""
    id: Optional[int] = None
    user_id: int = 0
    amount: Decimal = Decimal('0')
    description: str = ""
    category: str = ""
    transaction_date: Optional[date] = None
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        """Преобразует в словарь"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'amount': float(self.amount),
            'description': self.description,
            'category': self.category,
            'transaction_date': self.transaction_date.isoformat() if self.transaction_date else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Expense':
        """Создает из словаря"""
        return cls(
            id=data.get('id'),
            user_id=data['user_id'],
            amount=Decimal(str(data['amount'])),
            description=data['description'],
            category=data['category'],
            transaction_date=datetime.fromisoformat(data['transaction_date']).date() if data.get('transaction_date') else None,
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None
        )
