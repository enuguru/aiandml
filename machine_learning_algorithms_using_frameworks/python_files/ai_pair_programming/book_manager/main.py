from services.book_service import BookService

def main():
    service = BookService("books_data.txt")

    while True:
        print("\n1. Add Book\n2. List Books\n3. Search by Author\n4. Remove Book\n5. Exit")
        choice = input("Choose: ")

        if choice == "1":
            title = input("Title: ")
            author = input("Author: ")
            year = int(input("Year: "))
            service.add_book(title, author, year)
            print("✅ Book added.")

        elif choice == "2":
            books = service.list_books()
            for book in books:
                print(f"📘 {book}")

        elif choice == "3":
            author = input("Author: ")
            results = service.find_books_by_author(author)
            for book in results:
                print(f"🔍 {book}")

        elif choice == "4":
            title = input("Title to remove: ")
            service.remove_book(title)
            print("🗑️ Book removed.")

        elif choice == "5":
            break

        else:
            print("❌ Invalid choice. Try again.")

if __name__ == "__main__":
    main()
