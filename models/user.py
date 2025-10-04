"""
Модель пользователя
"""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

class UserRole(Enum):
    """Роли пользователей"""
    USER = "user"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

@dataclass
class User:
    """Модель пользователя"""
    id: int
    username: str
    phone: Optional[str] = None
    role: UserRole = UserRole.USER
    is_active: bool = True
    created_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    folder_name: Optional[str] = None
    
    def is_admin(self) -> bool:
        """Проверяет, является ли пользователь администратором"""
        return self.role in [UserRole.ADMIN, UserRole.SUPER_ADMIN]
    
    def is_super_admin(self) -> bool:
        """Проверяет, является ли пользователь супер-администратором"""
        return self.role == UserRole.SUPER_ADMIN
    
    def to_dict(self) -> dict:
        """Преобразует в словарь"""
        return {
            'id': self.id,
            'username': self.username,
            'phone': self.phone,
            'role': self.role.value,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_activity': self.last_activity.isoformat() if self.last_activity else None,
            'folder_name': self.folder_name
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'User':
        """Создает из словаря"""
        return cls(
            id=data['id'],
            username=data['username'],
            phone=data.get('phone'),
            role=UserRole(data.get('role', 'user')),
            is_active=data.get('is_active', True),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None,
            last_activity=datetime.fromisoformat(data['last_activity']) if data.get('last_activity') else None,
            folder_name=data.get('folder_name')
        )
