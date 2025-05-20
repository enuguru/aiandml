from models.book import Book
from utils.file_handler import read_books, write_books

class BookService:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.books = read_books(storage_path)

    def add_book(self, title: str, author: str, year: int):
        book = Book(title, author, year)
        self.books.append(book)
        write_books(self.books, self.storage_path)

    def list_books(self):
        return self.books

    def find_books_by_author(self, author: str):
        return [book for book in self.books if book.author.lower() == author.lower()]

    def remove_book(self, title: str):
        self.books = [book for book in self.books if book.title.lower() != title.lower()]
        write_books(self.books, self.storage_path)
