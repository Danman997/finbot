"""
Тесты для сервисов
"""
import pytest
import asyncio
from decimal import Decimal
from datetime import date, datetime
from services.user_service import user_service
from services.expense_service import expense_service
from services.budget_service import budget_service
from services.reminder_service import reminder_service
from services.group_service import group_service
from services.classification_service import classification_service
from models.user import UserRole
from models.budget_plan import BudgetPlanItem

@pytest.mark.asyncio
async def test_user_service():
    """Тест сервиса пользователей"""
    # Создание пользователя
    user = await user_service.create_user(
        user_id=12345,
        username="test_user",
        phone="+1234567890",
        role=UserRole.USER
    )
    
    assert user.id == 12345
    assert user.username == "test_user"
    assert user.phone == "+1234567890"
    assert user.role == UserRole.USER
    
    # Получение пользователя
    retrieved_user = await user_service.get_user(12345)
    assert retrieved_user is not None
    assert retrieved_user.username == "test_user"
    
    # Обновление пользователя
    updated_user = await user_service.update_user(12345, username="updated_user")
    assert updated_user is not None
    assert updated_user.username == "updated_user"
    
    # Проверка существования
    exists = await user_service.is_user_exists(12345)
    assert exists is True
    
    # Удаление пользователя
    deleted = await user_service.delete_user(12345)
    assert deleted is True

@pytest.mark.asyncio
async def test_expense_service():
    """Тест сервиса расходов"""
    # Создание пользователя для теста
    user = await user_service.create_user(
        user_id=12346,
        username="expense_test_user"
    )
    
    # Создание расхода
    expense = await expense_service.create_expense(
        user_id=12346,
        amount=Decimal('1500.50'),
        description="Тестовый расход",
        category="Продукты",
        transaction_date=date.today()
    )
    
    assert expense.user_id == 12346
    assert expense.amount == Decimal('1500.50')
    assert expense.description == "Тестовый расход"
    assert expense.category == "Продукты"
    
    # Получение расхода
    retrieved_expense = await expense_service.get_expense(expense.id)
    assert retrieved_expense is not None
    assert retrieved_expense.description == "Тестовый расход"
    
    # Обновление расхода
    updated_expense = await expense_service.update_expense(
        expense.id, 
        description="Обновленный расход"
    )
    assert updated_expense is not None
    assert updated_expense.description == "Обновленный расход"
    
    # Получение расходов пользователя
    user_expenses = await expense_service.get_user_expenses(12346)
    assert len(user_expenses) == 1
    assert user_expenses[0].description == "Обновленный расход"
    
    # Получение сводки
    summary = await expense_service.get_expenses_summary(12346)
    assert summary['total_amount'] == Decimal('1500.50')
    assert summary['total_count'] == 1
    
    # Удаление расхода
    deleted = await expense_service.delete_expense(expense.id)
    assert deleted is True
    
    # Очистка
    await user_service.delete_user(12346)

@pytest.mark.asyncio
async def test_budget_service():
    """Тест сервиса планирования бюджета"""
    # Создание пользователя для теста
    user = await user_service.create_user(
        user_id=12347,
        username="budget_test_user"
    )
    
    # Создание плана
    plan_date = date(2025, 9, 1)
    items = [
        BudgetPlanItem(category="Продукты", amount=Decimal('50000')),
        BudgetPlanItem(category="Транспорт", amount=Decimal('30000'))
    ]
    
    plan = await budget_service.create_budget_plan(
        user_id=12347,
        plan_month=plan_date,
        total_amount=Decimal('80000'),
        items=items
    )
    
    assert plan.user_id == 12347
    assert plan.plan_month == plan_date
    assert plan.total_amount == Decimal('80000')
    assert len(plan.items) == 2
    
    # Получение плана
    retrieved_plan = await budget_service.get_budget_plan(plan.id)
    assert retrieved_plan is not None
    assert retrieved_plan.total_amount == Decimal('80000')
    
    # Получение плана по месяцу
    month_plan = await budget_service.get_budget_plan_by_month(12347, 9, 2025)
    assert month_plan is not None
    assert month_plan.id == plan.id
    
    # Обновление плана
    updated_plan = await budget_service.update_budget_plan(
        plan.id,
        total_amount=Decimal('90000')
    )
    assert updated_plan is not None
    assert updated_plan.total_amount == Decimal('90000')
    
    # Удаление плана
    deleted = await budget_service.delete_budget_plan(plan.id)
    assert deleted is True
    
    # Очистка
    await user_service.delete_user(12347)

@pytest.mark.asyncio
async def test_reminder_service():
    """Тест сервиса напоминаний"""
    # Создание пользователя для теста
    user = await user_service.create_user(
        user_id=12348,
        username="reminder_test_user"
    )
    
    # Создание напоминания
    reminder = await reminder_service.create_reminder(
        user_id=12348,
        title="Тестовое напоминание",
        description="Описание напоминания",
        amount=Decimal('10000'),
        start_date=date.today(),
        end_date=date(2025, 12, 31)
    )
    
    assert reminder.user_id == 12348
    assert reminder.title == "Тестовое напоминание"
    assert reminder.amount == Decimal('10000')
    assert reminder.is_active is True
    
    # Получение напоминания
    retrieved_reminder = await reminder_service.get_reminder(reminder.id)
    assert retrieved_reminder is not None
    assert retrieved_reminder.title == "Тестовое напоминание"
    
    # Обновление напоминания
    updated_reminder = await reminder_service.update_reminder(
        reminder.id,
        title="Обновленное напоминание"
    )
    assert updated_reminder is not None
    assert updated_reminder.title == "Обновленное напоминание"
    
    # Получение напоминаний пользователя
    user_reminders = await reminder_service.get_user_reminders(12348)
    assert len(user_reminders) == 1
    assert user_reminders[0].title == "Обновленное напоминание"
    
    # Удаление напоминания
    deleted = await reminder_service.delete_reminder(reminder.id)
    assert deleted is True
    
    # Очистка
    await user_service.delete_user(12348)

@pytest.mark.asyncio
async def test_group_service():
    """Тест сервиса групп"""
    # Создание пользователей для теста
    admin_user = await user_service.create_user(
        user_id=12349,
        username="group_admin"
    )
    
    member_user = await user_service.create_user(
        user_id=12350,
        username="group_member"
    )
    
    # Создание группы
    group = await group_service.create_group("Тестовая группа", 12349)
    
    assert group.name == "Тестовая группа"
    assert group.admin_user_id == 12349
    assert group.invitation_code is not None
    
    # Получение группы
    retrieved_group = await group_service.get_group(group.id)
    assert retrieved_group is not None
    assert retrieved_group.name == "Тестовая группа"
    
    # Присоединение участника
    member = await group_service.add_member(group.id, 12350, "member")
    assert member.group_id == group.id
    assert member.user_id == 12350
    assert member.role == "member"
    
    # Получение участников группы
    members = await group_service.get_group_members(group.id)
    assert len(members) == 2  # Администратор + участник
    
    # Получение групп пользователя
    user_groups = await group_service.get_user_groups(12350)
    assert len(user_groups) == 1
    assert user_groups[0].id == group.id
    
    # Удаление участника
    removed = await group_service.remove_member(group.id, 12350)
    assert removed is True
    
    # Удаление группы
    deleted = await group_service.delete_group(group.id)
    assert deleted is True
    
    # Очистка
    await user_service.delete_user(12349)
    await user_service.delete_user(12350)

def test_classification_service():
    """Тест сервиса классификации"""
    # Обучение модели
    classification_service.train_model()
    
    # Тестирование классификации
    test_cases = [
        ("хлеб молоко", "Продукты"),
        ("бензин заправка", "Транспорт"),
        ("кино билет", "Развлечения"),
        ("врач больница", "Здоровье"),
        ("футболка рубашка", "Одежда")
    ]
    
    for description, expected_category in test_cases:
        predicted_category = classification_service.classify_expense(description)
        assert predicted_category == expected_category, f"Ожидалось {expected_category}, получено {predicted_category} для '{description}'"
    
    # Тестирование уверенности
    confidence = classification_service.get_classification_confidence("хлеб молоко")
    assert 0 <= confidence <= 1

if __name__ == "__main__":
    # Запуск тестов
    asyncio.run(pytest.main([__file__, "-v"]))
