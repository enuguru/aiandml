from models.employee import Employee

def calculate_tax(employee):
    if employee.base_salary < 50000:
        return employee.base_salary * 0.10
    else:
        return employee.base_salary * 0.20

def compute_net_salary(employee):
    tax = calculate_tax(employee)
    return employee.base_salary - tax

def generate_salary_report(employees):
    report = []
    for emp in employees:
        net_salary = compute_net_salary(emp)
        report.append(f"{emp.name} ({emp.emp_id}) - Net Salary: {net_salary}")
    return "\n".join(report)

def group_by_department(employees):
    dept_map = {}
    for emp in employees:
        dept_map.setdefault(emp.department, []).append(emp)
    return dept_map
