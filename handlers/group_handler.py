"""
Обработчик групп
"""
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import ContextTypes
from typing import Optional, List
from handlers.base_handler import BaseHandler
from services.group_service import group_service
from services.user_service import user_service
from utils import logger, ValidationError, DatabaseError
from utils.validators import Validator

class GroupHandler(BaseHandler):
    """Обработчик групп"""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Основной метод обработки"""
        if not await self.check_user_access(update, context):
            return
        
        try:
            text = update.message.text.strip()
            user_id = update.effective_user.id
            
            # Обработка различных команд
            if text == "👥 Группы":
                await self._show_groups_menu(update, context, user_id)
            elif text == "➕ Создать группу":
                await self._start_group_creation(update, context, user_id)
            elif text == "🔗 Присоединиться к группе":
                await self._start_group_joining(update, context, user_id)
            elif text == "📋 Мои группы":
                await self._show_user_groups(update, context, user_id)
            else:
                await update.message.reply_text(
                    "❌ Неизвестная команда",
                    reply_markup=self.get_main_menu_keyboard()
                )
                
        except Exception as e:
            await self.handle_error(update, context, e)
    
    async def _show_groups_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE, user_id: int):
        """Показ меню групп"""
        try:
            # Получаем группы пользователя
            user_groups = await group_service.get_user_groups(user_id)
            
            if not user_groups:
                keyboard = [
                    [KeyboardButton("➕ Создать группу"), KeyboardButton("🔗 Присоединиться к группе")],
                    [KeyboardButton("🔙 Назад")]
                ]
                message = "👥 Группы\n\nУ вас пока нет групп."
            else:
                keyboard = [
                    [KeyboardButton("➕ Создать группу"), KeyboardButton("🔗 Присоединиться к группе")],
                    [KeyboardButton("📋 Мои группы")],
                    [KeyboardButton("🔙 Назад")]
                ]
                
                message = "👥 Группы\n\n"
                message += "📋 Ваши группы:\n"
                for group in user_groups[:3]:  # Показываем последние 3 группы
                    message += f"• {group.name}\n"
                
                if len(user_groups) > 3:
                    message += f"... и еще {len(user_groups) - 3} групп"
            
            await update.message.reply_text(
                message,
                reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
            )
            
        except Exception as e:
            await self.handle_error(update, context, e)
    
    async def _start_group_creation(self, update: Update, context: ContextTypes.DEFAULT_TYPE, user_id: int):
        """Начало создания группы"""
        try:
            context.user_data['group_user_id'] = user_id
            context.user_data['group_step'] = 'name'
            
            await update.message.reply_text(
                "➕ Создание группы\n\n"
                "Введите название группы:",
                reply_markup=self.get_back_keyboard()
            )
            
        except Exception as e:
            await self.handle_error(update, context, e)
    
    async def _process_group_name(self, update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
        """Обработка ввода названия группы"""
        try:
            # Валидируем название группы
            name = Validator.validate_length(text, 2, 50, "Название группы")
            
            # Создаем группу
            group = await group_service.create_group(name, context.user_data['group_user_id'])
            
            # Формируем ответ
            response = f"✅ Группа создана!\n\n"
            response += f"👥 Название: {group.name}\n"
            response += f"🔑 Код приглашения: {group.invitation_code}\n\n"
            response += f"Поделитесь этим кодом с другими пользователями для присоединения к группе."
            
            await update.message.reply_text(
                response,
                reply_markup=self.get_main_menu_keyboard()
            )
            
            # Очищаем данные
            self._clear_group_data(context)
            
        except ValidationError as e:
            await update.message.reply_text(
                f"❌ {e.message}",
                reply_markup=self.get_back_keyboard()
            )
        except Exception as e:
            await self.handle_error(update, context, e)
    
    async def _start_group_joining(self, update: Update, context: ContextTypes.DEFAULT_TYPE, user_id: int):
        """Начало присоединения к группе"""
        try:
            context.user_data['group_user_id'] = user_id
            context.user_data['group_step'] = 'invitation_code'
            
            await update.message.reply_text(
                "🔗 Присоединение к группе\n\n"
                "Введите код приглашения:",
                reply_markup=self.get_back_keyboard()
            )
            
        except Exception as e:
            await self.handle_error(update, context, e)
    
    async def _process_invitation_code(self, update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
        """Обработка ввода кода приглашения"""
        try:
            # Валидируем код приглашения
            invitation_code = Validator.validate_invitation_code(text)
            
            # Получаем пользователя
            user = await user_service.get_user(context.user_data['group_user_id'])
            if not user:
                await update.message.reply_text(
                    "❌ Пользователь не найден",
                    reply_markup=self.get_back_keyboard()
                )
                return
            
            # Присоединяемся к группе
            success, message = await group_service.join_group_by_invitation(
                invitation_code, user.id, user.phone or ""
            )
            
            if success:
                # Получаем информацию о группе
                group = await group_service.get_group_by_invitation_code(invitation_code)
                
                response = f"✅ Успешно присоединились к группе!\n\n"
                response += f"👥 Группа: {group.name if group else 'Неизвестно'}\n"
                response += f"🔑 Код: {invitation_code}"
                
                await update.message.reply_text(
                    response,
                    reply_markup=self.get_main_menu_keyboard()
                )
            else:
                await update.message.reply_text(
                    f"❌ {message}",
                    reply_markup=self.get_back_keyboard()
                )
            
            # Очищаем данные
            self._clear_group_data(context)
            
        except ValidationError as e:
            await update.message.reply_text(
                f"❌ {e.message}",
                reply_markup=self.get_back_keyboard()
            )
        except Exception as e:
            await self.handle_error(update, context, e)
    
    async def _show_user_groups(self, update: Update, context: ContextTypes.DEFAULT_TYPE, user_id: int):
        """Показ групп пользователя"""
        try:
            # Получаем группы пользователя
            user_groups = await group_service.get_user_groups(user_id)
            
            if not user_groups:
                await update.message.reply_text(
                    "📋 У вас пока нет групп.",
                    reply_markup=self.get_back_keyboard()
                )
                return
            
            # Формируем список
            response = "📋 Ваши группы:\n\n"
            
            for i, group in enumerate(user_groups, 1):
                # Получаем количество участников
                members_count = await group_service.get_group_members_count(group.id)
                
                response += f"{i}. {group.name}\n"
                response += f"   👥 Участников: {members_count}/5\n"
                response += f"   🔑 Код: {group.invitation_code}\n"
                response += f"   📅 Создана: {self.format_date(group.created_at)}\n\n"
            
            await update.message.reply_text(
                response,
                reply_markup=self.get_back_keyboard()
            )
            
        except Exception as e:
            await self.handle_error(update, context, e)
    
    async def _show_group_members(self, update: Update, context: ContextTypes.DEFAULT_TYPE, group_id: int):
        """Показ участников группы"""
        try:
            # Получаем участников группы
            members = await group_service.get_group_members(group_id)
            
            if not members:
                await update.message.reply_text(
                    "👥 В группе нет участников",
                    reply_markup=self.get_back_keyboard()
                )
                return
            
            # Формируем список
            response = "👥 Участники группы:\n\n"
            
            for i, member in enumerate(members, 1):
                # Получаем информацию о пользователе
                user = await user_service.get_user(member.user_id)
                username = user.username if user else f"Пользователь {member.user_id}"
                
                response += f"{i}. {username}\n"
                response += f"   👨‍💼 Роль: {member.role}\n"
                response += f"   📅 Присоединился: {self.format_date(member.joined_at)}\n\n"
            
            await update.message.reply_text(
                response,
                reply_markup=self.get_back_keyboard()
            )
            
        except Exception as e:
            await self.handle_error(update, context, e)
    
    def _clear_group_data(self, context: ContextTypes.DEFAULT_TYPE):
        """Очистка данных групп"""
        keys_to_remove = [
            'group_user_id', 'group_step'
        ]
        for key in keys_to_remove:
            context.user_data.pop(key, None)
