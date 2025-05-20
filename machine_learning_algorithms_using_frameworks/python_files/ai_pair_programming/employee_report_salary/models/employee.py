class Employee:
    def __init__(self, emp_id, name, department, base_salary):
        self.emp_id = emp_id
        self.name = name
        self.department = department
        self.base_salary = base_salary

    def __str__(self):
        return f"{self.name} ({self.emp_id}) - {self.department}: {self.base_salary}"
