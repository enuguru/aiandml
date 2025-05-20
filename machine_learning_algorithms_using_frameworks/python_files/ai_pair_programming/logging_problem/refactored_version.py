import logging
from collections import Counter
from typing import List, Tuple

logging.basicConfig(level=logging.INFO)

class LogAnalyzer:
    """
    A class to read, parse, and analyze error messages from log files.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.errors: List[Tuple[str, str]] = []

    def read_logs(self) -> List[str]:
        """Reads all lines from the log file."""
        try:
            with open(self.filepath, 'r') as file:
                lines = file.readlines()
            logging.info("Log file loaded.")
            return lines
        except FileNotFoundError:
            logging.error("Log file not found.")
            return []

    def extract_errors(self, lines: List[str]) -> None:
        """Parses lines and extracts error entries with timestamps."""
        for line in lines:
            if "ERROR" in line:
                parts = line.strip().split(" - ")
                if len(parts) >= 3:
                    timestamp = parts[0]
                    message = parts[2]
                    self.errors.append((timestamp, message))
        logging.info(f"Found {len(self.errors)} errors.")

    def count_error_types(self) -> Counter:
        """Counts occurrences of each unique error message."""
        messages = [msg for _, msg in self.errors]
        return Counter(messages)

    def display_top_errors(self, top_n: int = 5) -> None:
        """Displays the most common error types."""
        counter = self.count_error_types()
        print(f"\nTop {top_n} Error Types:")
        for i, (error, count) in enumerate(counter.most_common(top_n), 1):
            print(f"{i}. {error} - {count} times")

def main():
    analyzer = LogAnalyzer("application.log")
    lines = analyzer.read_logs()
    analyzer.extract_errors(lines)
    analyzer.display_top_errors()

if __name__ == "__main__":
    main()
