"""
Сервис для работы с группами
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
import secrets
import string
from models.group import Group, GroupMember
from services.database_service import db_service
from utils import logger, DatabaseError, ValidationError

class GroupService:
    """Сервис для работы с группами"""
    
    def _generate_invitation_code(self) -> str:
        """Генерация кода приглашения"""
        return ''.join(secrets.choices(string.ascii_uppercase + string.digits, k=8))
    
    async def create_group(self, name: str, admin_user_id: int) -> Group:
        """Создание новой группы"""
        try:
            # Генерируем уникальный код приглашения
            invitation_code = self._generate_invitation_code()
            
            # Проверяем уникальность кода
            while await self.get_group_by_invitation_code(invitation_code):
                invitation_code = self._generate_invitation_code()
            
            query = """
                INSERT INTO groups (name, admin_user_id, invitation_code, created_at)
                VALUES ($1, $2, $3, $4)
                RETURNING *
            """
            
            now = datetime.now()
            result = await db_service.fetch_one(query, name, admin_user_id, invitation_code, now)
            
            if not result:
                raise DatabaseError("Не удалось создать группу")
            
            # Добавляем администратора в группу
            await self.add_member(result['id'], admin_user_id, "admin")
            
            logger.info(f"Создана группа: {name} (код: {invitation_code})")
            return Group.from_dict(dict(result))
            
        except Exception as e:
            logger.error(f"Ошибка создания группы: {e}")
            raise DatabaseError(f"Не удалось создать группу: {e}")
    
    async def get_group(self, group_id: int) -> Optional[Group]:
        """Получение группы по ID"""
        try:
            query = "SELECT * FROM groups WHERE id = $1"
            result = await db_service.fetch_one(query, group_id)
            
            if result:
                return Group.from_dict(dict(result))
            return None
            
        except Exception as e:
            logger.error(f"Ошибка получения группы {group_id}: {e}")
            raise DatabaseError(f"Не удалось получить группу: {e}")
    
    async def get_group_by_invitation_code(self, invitation_code: str) -> Optional[Group]:
        """Получение группы по коду приглашения"""
        try:
            query = "SELECT * FROM groups WHERE invitation_code = $1"
            result = await db_service.fetch_one(query, invitation_code)
            
            if result:
                return Group.from_dict(dict(result))
            return None
            
        except Exception as e:
            logger.error(f"Ошибка получения группы по коду {invitation_code}: {e}")
            raise DatabaseError(f"Не удалось получить группу: {e}")
    
    async def get_user_groups(self, user_id: int) -> List[Group]:
        """Получение групп пользователя"""
        try:
            query = """
                SELECT g.* FROM groups g
                INNER JOIN group_members gm ON g.id = gm.group_id
                WHERE gm.user_id = $1
                ORDER BY g.created_at DESC
            """
            results = await db_service.fetch_all(query, user_id)
            
            return [Group.from_dict(dict(row)) for row in results]
            
        except Exception as e:
            logger.error(f"Ошибка получения групп пользователя {user_id}: {e}")
            raise DatabaseError(f"Не удалось получить группы: {e}")
    
    async def add_member(self, group_id: int, user_id: int, role: str = "member") -> GroupMember:
        """Добавление участника в группу"""
        try:
            # Проверяем, не является ли пользователь уже участником
            existing_member = await self.get_group_member(group_id, user_id)
            if existing_member:
                raise ValidationError("Пользователь уже является участником группы")
            
            # Проверяем лимит участников
            members_count = await self.get_group_members_count(group_id)
            if members_count >= 5:  # Максимум 5 участников
                raise ValidationError("Группа уже заполнена (максимум 5 участников)")
            
            query = """
                INSERT INTO group_members (group_id, user_id, role, joined_at)
                VALUES ($1, $2, $3, $4)
                RETURNING *
            """
            
            now = datetime.now()
            result = await db_service.fetch_one(query, group_id, user_id, role, now)
            
            if not result:
                raise DatabaseError("Не удалось добавить участника")
            
            logger.info(f"Добавлен участник {user_id} в группу {group_id}")
            return GroupMember.from_dict(dict(result))
            
        except Exception as e:
            logger.error(f"Ошибка добавления участника: {e}")
            raise DatabaseError(f"Не удалось добавить участника: {e}")
    
    async def remove_member(self, group_id: int, user_id: int) -> bool:
        """Удаление участника из группы"""
        try:
            query = "DELETE FROM group_members WHERE group_id = $1 AND user_id = $2"
            result = await db_service.execute(query, group_id, user_id)
            
            if "DELETE 1" in result:
                logger.info(f"Удален участник {user_id} из группы {group_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Ошибка удаления участника: {e}")
            raise DatabaseError(f"Не удалось удалить участника: {e}")
    
    async def get_group_member(self, group_id: int, user_id: int) -> Optional[GroupMember]:
        """Получение участника группы"""
        try:
            query = "SELECT * FROM group_members WHERE group_id = $1 AND user_id = $2"
            result = await db_service.fetch_one(query, group_id, user_id)
            
            if result:
                return GroupMember.from_dict(dict(result))
            return None
            
        except Exception as e:
            logger.error(f"Ошибка получения участника группы: {e}")
            raise DatabaseError(f"Не удалось получить участника группы: {e}")
    
    async def get_group_members(self, group_id: int) -> List[GroupMember]:
        """Получение всех участников группы"""
        try:
            query = """
                SELECT * FROM group_members 
                WHERE group_id = $1 
                ORDER BY joined_at ASC
            """
            results = await db_service.fetch_all(query, group_id)
            
            return [GroupMember.from_dict(dict(row)) for row in results]
            
        except Exception as e:
            logger.error(f"Ошибка получения участников группы {group_id}: {e}")
            raise DatabaseError(f"Не удалось получить участников группы: {e}")
    
    async def get_group_members_count(self, group_id: int) -> int:
        """Получение количества участников группы"""
        try:
            query = "SELECT COUNT(*) FROM group_members WHERE group_id = $1"
            return await db_service.fetch_val(query, group_id)
            
        except Exception as e:
            logger.error(f"Ошибка получения количества участников группы {group_id}: {e}")
            return 0
    
    async def update_member_role(self, group_id: int, user_id: int, new_role: str) -> Optional[GroupMember]:
        """Обновление роли участника"""
        try:
            query = """
                UPDATE group_members 
                SET role = $1
                WHERE group_id = $2 AND user_id = $3
                RETURNING *
            """
            
            result = await db_service.fetch_one(query, new_role, group_id, user_id)
            
            if result:
                logger.info(f"Обновлена роль участника {user_id} в группе {group_id}")
                return GroupMember.from_dict(dict(result))
            return None
            
        except Exception as e:
            logger.error(f"Ошибка обновления роли участника: {e}")
            raise DatabaseError(f"Не удалось обновить роль участника: {e}")
    
    async def delete_group(self, group_id: int) -> bool:
        """Удаление группы"""
        try:
            async with db_service.transaction() as conn:
                # Удаляем участников
                delete_members_query = "DELETE FROM group_members WHERE group_id = $1"
                await conn.execute(delete_members_query, group_id)
                
                # Удаляем группу
                delete_group_query = "DELETE FROM groups WHERE id = $1"
                result = await conn.execute(delete_group_query, group_id)
                
                if "DELETE 1" in result:
                    logger.info(f"Удалена группа: {group_id}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Ошибка удаления группы {group_id}: {e}")
            raise DatabaseError(f"Не удалось удалить группу: {e}")
    
    async def get_groups_summary(self) -> Dict[str, Any]:
        """Получение сводки по группам"""
        try:
            query = """
                SELECT 
                    COUNT(*) as total_groups,
                    AVG(member_count) as avg_members_per_group
                FROM (
                    SELECT g.id, COUNT(gm.user_id) as member_count
                    FROM groups g
                    LEFT JOIN group_members gm ON g.id = gm.group_id
                    GROUP BY g.id
                ) group_stats
            """
            
            result = await db_service.fetch_one(query)
            
            if result:
                return {
                    'total_groups': result['total_groups'],
                    'avg_members_per_group': float(result['avg_members_per_group'] or 0)
                }
            
            return {
                'total_groups': 0,
                'avg_members_per_group': 0
            }
            
        except Exception as e:
            logger.error(f"Ошибка получения сводки групп: {e}")
            raise DatabaseError(f"Не удалось получить сводку групп: {e}")

# Глобальный экземпляр сервиса
group_service = GroupService()
