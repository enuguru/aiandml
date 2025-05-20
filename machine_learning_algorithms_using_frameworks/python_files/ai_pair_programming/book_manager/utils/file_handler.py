from models.book import Book

def read_books(filepath: str):
    books = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                title, author, year = line.strip().split('|')
                books.append(Book(title, author, int(year)))
    except FileNotFoundError:
        pass
    return books

def write_books(books, filepath: str):
    with open(filepath, 'w') as f:
        for book in books:
            f.write(f"{book.title}|{book.author}|{book.year}\n")
