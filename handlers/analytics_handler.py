"""
Обработчик аналитики
"""
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import ContextTypes
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Any
from handlers.base_handler import BaseHandler
from services.expense_service import expense_service
from services.budget_service import budget_service
from utils import logger, DatabaseError

class AnalyticsHandler(BaseHandler):
    """Обработчик аналитики"""
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Основной метод обработки"""
        if not await self.check_user_access(update, context):
            return
        
        try:
            text = update.message.text.strip()
            user_id = update.effective_user.id
            
            # Обработка различных команд
            if text == "📈 Аналитика":
                await self._show_analytics_menu(update, context, user_id)
            elif text == "📊 Сравнение с планом":
                await self._show_plan_comparison(update, context, user_id)
            elif text in ["Сегодня", "Неделя", "Месяц", "Год"]:
                await self._show_period_analytics(update, context, user_id, text)
            else:
                await update.message.reply_text(
                    "❌ Неизвестная команда",
                    reply_markup=self.get_main_menu_keyboard()
                )
                
        except Exception as e:
            await self.handle_error(update, context, e)
    
    async def _show_analytics_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE, user_id: int):
        """Показ меню аналитики"""
        try:
            keyboard = [
                [KeyboardButton("📊 Сравнение с планом")],
                [KeyboardButton("Сегодня"), KeyboardButton("Неделя")],
                [KeyboardButton("Месяц"), KeyboardButton("Год")],
                [KeyboardButton("🔙 Назад")]
            ]
            
            await update.message.reply_text(
                "📈 Аналитика\n\n"
                "Выберите тип анализа:",
                reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
            )
            
        except Exception as e:
            await self.handle_error(update, context, e)
    
    async def _show_plan_comparison(self, update: Update, context: ContextTypes.DEFAULT_TYPE, user_id: int):
        """Показ сравнения с планом"""
        try:
            # Получаем текущий месяц
            today = date.today()
            current_month = today.month
            current_year = today.year
            
            # Получаем план на текущий месяц
            plan = await budget_service.get_budget_plan_by_month(user_id, current_month, current_year)
            
            if not plan:
                await update.message.reply_text(
                    f"📊 Сравнение с планом\n\n"
                    f"У вас нет плана на {current_month:02d}.{current_year}.\n"
                    f"Создайте план в разделе '📅 Планирование'.",
                    reply_markup=self.get_back_keyboard()
                )
                return
            
            # Получаем фактические расходы за текущий месяц
            start_date = date(current_year, current_month, 1)
            end_date = today
            
            expenses_summary = await expense_service.get_expenses_summary(user_id, start_date, end_date)
            
            # Формируем сравнение
            response = f"📊 Сравнение с планом за {current_month:02d}.{current_year}\n\n"
            
            # Общее сравнение
            planned_total = float(plan.total_amount)
            actual_total = float(expenses_summary['total_amount'])
            difference = actual_total - planned_total
            
            response += f"💰 Планировалось: {self.format_amount(planned_total)}\n"
            response += f"💸 Потрачено: {self.format_amount(actual_total)}\n"
            response += f"📈 Разница: {self.format_amount(difference)}\n\n"
            
            if difference > 0:
                response += f"⚠️ Превышение плана на {self.format_amount(difference)}\n"
            elif difference < 0:
                response += f"✅ Экономия: {self.format_amount(abs(difference))}\n"
            else:
                response += f"✅ План выполнен точно\n"
            
            # Сравнение по категориям
            response += "\n📋 По категориям:\n"
            
            # Создаем словарь плановых сумм
            planned_categories = {item.category: float(item.amount) for item in plan.items}
            
            # Сравниваем с фактическими расходами
            for category_data in expenses_summary['categories']:
                category = category_data['category']
                actual_amount = float(category_data['total_amount'])
                planned_amount = planned_categories.get(category, 0)
                
                response += f"\n🏷️ {category}:\n"
                response += f"   План: {self.format_amount(planned_amount)}\n"
                response += f"   Факт: {self.format_amount(actual_amount)}\n"
                
                if planned_amount > 0:
                    percentage = (actual_amount / planned_amount) * 100
                    response += f"   Выполнение: {percentage:.1f}%\n"
                    
                    if percentage > 100:
                        response += f"   ⚠️ Превышение на {self.format_amount(actual_amount - planned_amount)}\n"
                    elif percentage < 80:
                        response += f"   ✅ Экономия: {self.format_amount(planned_amount - actual_amount)}\n"
            
            await update.message.reply_text(
                response,
                reply_markup=self.get_back_keyboard()
            )
            
        except Exception as e:
            await self.handle_error(update, context, e)
    
    async def _show_period_analytics(self, update: Update, context: ContextTypes.DEFAULT_TYPE, user_id: int, period: str):
        """Показ аналитики за период"""
        try:
            # Определяем период
            end_date = date.today()
            if period == "Сегодня":
                start_date = end_date
                period_name = "сегодня"
            elif period == "Неделя":
                start_date = end_date - timedelta(days=7)
                period_name = "неделю"
            elif period == "Месяц":
                start_date = end_date.replace(day=1)
                period_name = "месяц"
            elif period == "Год":
                start_date = end_date.replace(month=1, day=1)
                period_name = "год"
            else:
                start_date = end_date.replace(day=1)
                period_name = "месяц"
            
            # Получаем сводку расходов
            expenses_summary = await expense_service.get_expenses_summary(user_id, start_date, end_date)
            
            if not expenses_summary['categories']:
                await update.message.reply_text(
                    f"📊 Аналитика за {period_name}\n\n"
                    f"За выбранный период нет расходов.",
                    reply_markup=self.get_back_keyboard()
                )
                return
            
            # Формируем аналитику
            response = f"📊 Аналитика за {period_name}\n\n"
            response += f"📅 Период: {self.format_date(start_date)} - {self.format_date(end_date)}\n\n"
            
            # Общая статистика
            total_amount = float(expenses_summary['total_amount'])
            total_count = expenses_summary['total_count']
            
            response += f"💰 Общая сумма: {self.format_amount(total_amount)}\n"
            response += f"📈 Количество транзакций: {total_count}\n"
            response += f"📊 Средний чек: {self.format_amount(total_amount / total_count if total_count > 0 else 0)}\n\n"
            
            # Топ категорий
            response += "🏆 Топ категорий:\n"
            for i, category_data in enumerate(expenses_summary['categories'][:5], 1):
                category = category_data['category']
                amount = float(category_data['total_amount'])
                percentage = (amount / total_amount) * 100 if total_amount > 0 else 0
                
                response += f"{i}. {category}: {self.format_amount(amount)} ({percentage:.1f}%)\n"
            
            # Дополнительная статистика
            if period == "Месяц":
                # Сравнение с предыдущим месяцем
                prev_month = start_date - timedelta(days=1)
                prev_start = prev_month.replace(day=1)
                prev_end = start_date - timedelta(days=1)
                
                try:
                    prev_summary = await expense_service.get_expenses_summary(user_id, prev_start, prev_end)
                    prev_total = float(prev_summary['total_amount'])
                    
                    if prev_total > 0:
                        change = ((total_amount - prev_total) / prev_total) * 100
                        response += f"\n📈 Изменение к прошлому месяцу: {change:+.1f}%\n"
                except:
                    pass
            
            await update.message.reply_text(
                response,
                reply_markup=self.get_back_keyboard()
            )
            
        except Exception as e:
            await self.handle_error(update, context, e)
    
    async def get_expense_trends(self, user_id: int, months: int = 6) -> str:
        """Получение трендов расходов"""
        try:
            # Получаем данные за последние месяцы
            today = date.today()
            trends = []
            
            for i in range(months):
                month_date = today.replace(day=1) - timedelta(days=30 * i)
                start_date = month_date.replace(day=1)
                
                # Определяем конец месяца
                if month_date.month == 12:
                    end_date = month_date.replace(year=month_date.year + 1, month=1, day=1) - timedelta(days=1)
                else:
                    end_date = month_date.replace(month=month_date.month + 1, day=1) - timedelta(days=1)
                
                # Получаем сводку за месяц
                summary = await expense_service.get_expenses_summary(user_id, start_date, end_date)
                total = float(summary['total_amount'])
                
                trends.append({
                    'month': month_date.strftime('%m.%Y'),
                    'total': total
                })
            
            # Формируем ответ
            response = "📈 Тренды расходов (последние 6 месяцев):\n\n"
            
            for trend in reversed(trends):  # Показываем от старых к новым
                response += f"{trend['month']}: {self.format_amount(trend['total'])}\n"
            
            # Вычисляем средний рост/снижение
            if len(trends) >= 2:
                recent = trends[0]['total']
                previous = trends[1]['total']
                
                if previous > 0:
                    change = ((recent - previous) / previous) * 100
                    if change > 0:
                        response += f"\n📈 Рост расходов: {change:+.1f}%"
                    elif change < 0:
                        response += f"\n📉 Снижение расходов: {change:+.1f}%"
                    else:
                        response += f"\n➡️ Расходы стабильны"
            
            return response
            
        except Exception as e:
            logger.error(f"Ошибка получения трендов: {e}")
            return "❌ Ошибка получения трендов расходов"
    
    async def get_category_analysis(self, user_id: int, category: str) -> str:
        """Анализ расходов по категории"""
        try:
            # Получаем расходы за последние 3 месяца
            today = date.today()
            start_date = today.replace(day=1) - timedelta(days=90)
            
            expenses = await expense_service.get_expenses_by_category(
                user_id, category, start_date, today
            )
            
            if not expenses:
                return f"📊 Нет расходов по категории '{category}' за последние 3 месяца"
            
            # Анализируем данные
            total_amount = sum(float(expense.amount) for expense in expenses)
            avg_amount = total_amount / len(expenses)
            
            # Группируем по месяцам
            monthly_data = {}
            for expense in expenses:
                month_key = expense.transaction_date.strftime('%m.%Y')
                if month_key not in monthly_data:
                    monthly_data[month_key] = {'amount': 0, 'count': 0}
                monthly_data[month_key]['amount'] += float(expense.amount)
                monthly_data[month_key]['count'] += 1
            
            # Формируем ответ
            response = f"📊 Анализ категории '{category}':\n\n"
            response += f"💰 Общая сумма: {self.format_amount(total_amount)}\n"
            response += f"📈 Количество транзакций: {len(expenses)}\n"
            response += f"📊 Средний чек: {self.format_amount(avg_amount)}\n\n"
            
            response += "📅 По месяцам:\n"
            for month, data in sorted(monthly_data.items()):
                response += f"{month}: {self.format_amount(data['amount'])} ({data['count']} транзакций)\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Ошибка анализа категории: {e}")
            return f"❌ Ошибка анализа категории '{category}'"
