"""
Модели групп
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class Group:
    """Модель группы"""
    id: Optional[int] = None
    name: str = ""
    admin_user_id: int = 0
    invitation_code: str = ""
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        """Преобразует в словарь"""
        return {
            'id': self.id,
            'name': self.name,
            'admin_user_id': self.admin_user_id,
            'invitation_code': self.invitation_code,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Group':
        """Создает из словаря"""
        return cls(
            id=data.get('id'),
            name=data['name'],
            admin_user_id=data['admin_user_id'],
            invitation_code=data['invitation_code'],
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None
        )

@dataclass
class GroupMember:
    """Модель участника группы"""
    id: Optional[int] = None
    group_id: int = 0
    user_id: int = 0
    role: str = "member"
    joined_at: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        """Преобразует в словарь"""
        return {
            'id': self.id,
            'group_id': self.group_id,
            'user_id': self.user_id,
            'role': self.role,
            'joined_at': self.joined_at.isoformat() if self.joined_at else None
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'GroupMember':
        """Создает из словаря"""
        return cls(
            id=data.get('id'),
            group_id=data['group_id'],
            user_id=data['user_id'],
            role=data.get('role', 'member'),
            joined_at=datetime.fromisoformat(data['joined_at']) if data.get('joined_at') else None
        )
