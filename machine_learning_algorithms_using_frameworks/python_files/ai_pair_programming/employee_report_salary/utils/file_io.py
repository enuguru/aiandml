def save_report(report, filename="salary_report.txt"):
    with open(filename, "w") as file:
        file.write(report)
