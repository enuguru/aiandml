def read_logs(path):
    with open(path) as f:
        lines = f.readlines()
    return lines

def parse_logs(lines):
    error_lines = []
    for line in lines:
        if "ERROR" in line:
            parts = line.strip().split(" - ")
            if len(parts) >= 3:
                timestamp = parts[0]
                level = parts[1]
                message = parts[2]
                error_lines.append((timestamp, message))
    return error_lines

def count_errors(errors):
    count_dict = {}
    for timestamp, message in errors:
        if message in count_dict:
            count_dict[message] += 1
        else:
            count_dict[message] = 1
    return count_dict

def display_top_errors(count_dict):
    sorted_errors = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
    for error, count in sorted_errors[:5]:
        print(f"{error} - {count} times")

def main():
    path = "application.log"
    lines = read_logs(path)
    errors = parse_logs(lines)
    count = count_errors(errors)
    display_top_errors(count)

main()
