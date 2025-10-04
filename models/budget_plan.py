"""
Модели планирования бюджета
"""
from dataclasses import dataclass
from datetime import datetime, date
from decimal import Decimal
from typing import Optional, List

@dataclass
class BudgetPlanItem:
    """Элемент плана бюджета"""
    id: Optional[int] = None
    plan_id: Optional[int] = None
    category: str = ""
    amount: Decimal = Decimal('0')
    comment: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Преобразует в словарь"""
        return {
            'id': self.id,
            'plan_id': self.plan_id,
            'category': self.category,
            'amount': float(self.amount),
            'comment': self.comment
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'BudgetPlanItem':
        """Создает из словаря"""
        return cls(
            id=data.get('id'),
            plan_id=data.get('plan_id'),
            category=data['category'],
            amount=Decimal(str(data['amount'])),
            comment=data.get('comment')
        )

@dataclass
class BudgetPlan:
    """План бюджета"""
    id: Optional[int] = None
    user_id: int = 0
    plan_month: Optional[date] = None
    total_amount: Decimal = Decimal('0')
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    items: List[BudgetPlanItem] = None
    
    def __post_init__(self):
        if self.items is None:
            self.items = []
    
    def get_total_amount(self) -> Decimal:
        """Возвращает общую сумму из элементов"""
        return sum(item.amount for item in self.items)
    
    def to_dict(self) -> dict:
        """Преобразует в словарь"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'plan_month': self.plan_month.isoformat() if self.plan_month else None,
            'total_amount': float(self.total_amount),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'items': [item.to_dict() for item in self.items]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'BudgetPlan':
        """Создает из словаря"""
        items = []
        if data.get('items'):
            items = [BudgetPlanItem.from_dict(item_data) for item_data in data['items']]
        
        return cls(
            id=data.get('id'),
            user_id=data['user_id'],
            plan_month=datetime.fromisoformat(data['plan_month']).date() if data.get('plan_month') else None,
            total_amount=Decimal(str(data['total_amount'])),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None,
            updated_at=datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else None,
            items=items
        )
