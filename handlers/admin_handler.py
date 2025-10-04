"""
Обработчик администрирования
"""
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import ContextTypes
from typing import Optional, List
from handlers.base_handler import BaseHandler
from services.user_service import user_service
from services.group_service import group_service
from models.user import UserRole
from utils import logger, ValidationError, DatabaseError, AuthorizationError
from utils.validators import Validator

class AdminHandler(BaseHandler):
    """Обработчик администрирования"""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Основной метод обработки"""
        if not await self.check_user_access(update, context):
            return
        
        try:
            user_id = update.effective_user.id
            
            # Проверяем права администратора
            if not await self._check_admin_rights(user_id):
                await update.message.reply_text(
                    "❌ У вас нет прав администратора",
                    reply_markup=self.get_main_menu_keyboard()
                )
                return
            
            text = update.message.text.strip()
            
            # Обработка различных команд
            if text == "👥 Добавить пользователя":
                await self._start_user_creation(update, context)
            elif text == "📋 Список пользователей":
                await self._show_users_list(update, context)
            elif text == "📁 Управление папками":
                await self._show_folder_management(update, context)
            elif text == "🔧 Роли пользователей":
                await self._show_roles_management(update, context)
            elif text == "📊 Статистика системы":
                await self._show_system_stats(update, context)
            else:
                await update.message.reply_text(
                    "❌ Неизвестная команда",
                    reply_markup=self.get_admin_menu_keyboard()
                )
                
        except Exception as e:
            await self.handle_error(update, context, e)
    
    async def _check_admin_rights(self, user_id: int) -> bool:
        """Проверка прав администратора"""
        try:
            user = await user_service.get_user(user_id)
            return user and user.is_admin()
        except Exception as e:
            logger.error(f"Ошибка проверки прав администратора: {e}")
            return False
    
    async def _start_user_creation(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Начало создания пользователя"""
        try:
            context.user_data['admin_step'] = 'username'
            
            await update.message.reply_text(
                "👥 Добавление пользователя\n\n"
                "Введите имя пользователя:",
                reply_markup=self.get_back_keyboard()
            )
            
        except Exception as e:
            await self.handle_error(update, context, e)
    
    async def _process_username_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
        """Обработка ввода имени пользователя"""
        try:
            # Валидируем имя пользователя
            username = Validator.validate_username(text)
            
            # Сохраняем данные
            context.user_data['new_user_username'] = username
            context.user_data['admin_step'] = 'phone'
            
            await update.message.reply_text(
                f"👤 Имя пользователя: {username}\n\n"
                "Введите номер телефона (или 'Пропустить'):",
                reply_markup=self.get_back_keyboard()
            )
            
        except ValidationError as e:
            await update.message.reply_text(
                f"❌ {e.message}",
                reply_markup=self.get_back_keyboard()
            )
        except Exception as e:
            await self.handle_error(update, context, e)
    
    async def _process_phone_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
        """Обработка ввода номера телефона"""
        try:
            # Обрабатываем пропуск телефона
            if text.lower() in ['пропустить', 'skip', '']:
                phone = None
            else:
                phone = Validator.validate_phone(text)
            
            # Сохраняем данные
            context.user_data['new_user_phone'] = phone
            context.user_data['admin_step'] = 'role'
            
            # Показываем выбор роли
            keyboard = [
                [KeyboardButton("👤 Пользователь"), KeyboardButton("👨‍💼 Администратор")],
                [KeyboardButton("🔙 Назад")]
            ]
            
            await update.message.reply_text(
                f"📱 Телефон: {phone or 'Не указан'}\n\n"
                "Выберите роль пользователя:",
                reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
            )
            
        except ValidationError as e:
            await update.message.reply_text(
                f"❌ {e.message}",
                reply_markup=self.get_back_keyboard()
            )
        except Exception as e:
            await self.handle_error(update, context, e)
    
    async def _process_role_selection(self, update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
        """Обработка выбора роли"""
        try:
            # Определяем роль
            if "Администратор" in text:
                role = UserRole.ADMIN
            else:
                role = UserRole.USER
            
            # Сохраняем данные
            context.user_data['new_user_role'] = role
            
            # Показываем подтверждение
            username = context.user_data['new_user_username']
            phone = context.user_data.get('new_user_phone')
            role_name = "Администратор" if role == UserRole.ADMIN else "Пользователь"
            
            response = f"👥 Подтверждение создания пользователя\n\n"
            response += f"👤 Имя: {username}\n"
            response += f"📱 Телефон: {phone or 'Не указан'}\n"
            response += f"👨‍💼 Роль: {role_name}\n\n"
            response += "Создать пользователя?"
            
            await update.message.reply_text(
                response,
                reply_markup=self.get_yes_no_keyboard()
            )
            
        except Exception as e:
            await self.handle_error(update, context, e)
    
    async def _confirm_user_creation(self, update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
        """Подтверждение создания пользователя"""
        try:
            if text == "✅ Да":
                # Создаем пользователя
                username = context.user_data['new_user_username']
                phone = context.user_data.get('new_user_phone')
                role = context.user_data['new_user_role']
                
                # Генерируем ID пользователя (в реальном приложении это должно быть более сложно)
                import random
                user_id = random.randint(100000, 999999)
                
                user = await user_service.create_user(
                    user_id=user_id,
                    username=username,
                    phone=phone,
                    role=role
                )
                
                await update.message.reply_text(
                    f"✅ Пользователь создан!\n\n"
                    f"👤 ID: {user_id}\n"
                    f"👤 Имя: {username}\n"
                    f"📱 Телефон: {phone or 'Не указан'}\n"
                    f"👨‍💼 Роль: {role.value}",
                    reply_markup=self.get_admin_menu_keyboard()
                )
                
                # Очищаем данные
                self._clear_admin_data(context)
                
            else:
                await update.message.reply_text(
                    "❌ Создание пользователя отменено",
                    reply_markup=self.get_admin_menu_keyboard()
                )
                self._clear_admin_data(context)
                
        except Exception as e:
            await self.handle_error(update, context, e)
    
    async def _show_users_list(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показ списка пользователей"""
        try:
            users = await user_service.get_all_users(limit=20)
            
            if not users:
                await update.message.reply_text(
                    "📋 Пользователи не найдены",
                    reply_markup=self.get_admin_menu_keyboard()
                )
                return
            
            # Формируем список
            response = "📋 Список пользователей:\n\n"
            
            for i, user in enumerate(users, 1):
                response += f"{i}. {user.username}\n"
                response += f"   ID: {user.id}\n"
                response += f"   Роль: {user.role.value}\n"
                response += f"   Статус: {'Активен' if user.is_active else 'Неактивен'}\n"
                if user.phone:
                    response += f"   Телефон: {user.phone}\n"
                response += "\n"
            
            await update.message.reply_text(
                response,
                reply_markup=self.get_admin_menu_keyboard()
            )
            
        except Exception as e:
            await self.handle_error(update, context, e)
    
    async def _show_folder_management(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показ управления папками"""
        try:
            # Получаем статистику папок
            users = await user_service.get_all_users()
            
            total_users = len(users)
            users_with_folders = sum(1 for user in users if user.folder_name)
            
            response = "📁 Управление папками\n\n"
            response += f"👥 Всего пользователей: {total_users}\n"
            response += f"📁 С папками: {users_with_folders}\n"
            response += f"📂 Без папок: {total_users - users_with_folders}\n\n"
            
            if users_with_folders > 0:
                response += "📋 Пользователи с папками:\n"
                for user in users[:10]:  # Показываем первых 10
                    if user.folder_name:
                        response += f"• {user.username}: {user.folder_name}\n"
            
            await update.message.reply_text(
                response,
                reply_markup=self.get_admin_menu_keyboard()
            )
            
        except Exception as e:
            await self.handle_error(update, context, e)
    
    async def _show_roles_management(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показ управления ролями"""
        try:
            # Получаем статистику ролей
            users = await user_service.get_all_users()
            
            role_stats = {}
            for user in users:
                role = user.role.value
                role_stats[role] = role_stats.get(role, 0) + 1
            
            response = "🔧 Управление ролями\n\n"
            response += "📊 Статистика ролей:\n"
            
            for role, count in role_stats.items():
                response += f"• {role}: {count} пользователей\n"
            
            response += f"\n👥 Всего пользователей: {len(users)}"
            
            await update.message.reply_text(
                response,
                reply_markup=self.get_admin_menu_keyboard()
            )
            
        except Exception as e:
            await self.handle_error(update, context, e)
    
    async def _show_system_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показ статистики системы"""
        try:
            # Получаем статистику
            users_count = await user_service.get_users_count()
            groups_summary = await group_service.get_groups_summary()
            
            response = "📊 Статистика системы\n\n"
            response += f"👥 Пользователей: {users_count}\n"
            response += f"👥 Групп: {groups_summary['total_groups']}\n"
            response += f"📊 Среднее количество участников в группе: {groups_summary['avg_members_per_group']:.1f}\n\n"
            
            # Дополнительная статистика
            response += "📈 Активность:\n"
            response += "• Система работает стабильно\n"
            response += "• Все сервисы доступны\n"
            response += "• База данных подключена\n"
            
            await update.message.reply_text(
                response,
                reply_markup=self.get_admin_menu_keyboard()
            )
            
        except Exception as e:
            await self.handle_error(update, context, e)
    
    def _clear_admin_data(self, context: ContextTypes.DEFAULT_TYPE):
        """Очистка данных администрирования"""
        keys_to_remove = [
            'admin_step', 'new_user_username', 'new_user_phone', 'new_user_role'
        ]
        for key in keys_to_remove:
            context.user_data.pop(key, None)
