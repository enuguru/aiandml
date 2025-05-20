# Case Study: Employee Salary Management System

## Objective:
Build a multi-file Python application using only built-in libraries that allows you to manage employee data, compute taxes, and generate salary reports.

## Functional Requirements:
1. Store employee data: name, ID, department, base salary
2. Calculate tax (e.g., 10% if salary < 50000, otherwise 20%)
3. Generate a formatted salary report with net pay
4. List all employees by department
5. Write report to a text file

## Files/Modules:
- models/employee.py: Employee class
- services/payroll.py: Business logic for tax and salary
- utils/file_io.py: Reading and writing reports
- main.py: Command-line interface to manage operations
