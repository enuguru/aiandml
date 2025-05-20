from models.employee import Employee
from services.payroll import generate_salary_report, group_by_department
from utils.file_io import save_report

def main():
    employees = [
        Employee("E001", "Alice", "Engineering", 72000),
        Employee("E002", "Bob", "Engineering", 45000),
        Employee("E003", "Charlie", "HR", 50000),
        Employee("E004", "David", "Marketing", 30000),
    ]

    print("Salary Report:")
    report = generate_salary_report(employees)
    print(report)

    print("\nEmployees by Department:")
    dept_map = group_by_department(employees)
    for dept, emps in dept_map.items():
        print(f"{dept}: {[e.name for e in emps]}")

    save_report(report)

if __name__ == "__main__":
    main()
