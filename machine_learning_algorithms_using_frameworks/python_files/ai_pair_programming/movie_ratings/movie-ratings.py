import csv

def read_file(file_path):
    with open(file_path, mode='r') as file:
        data = list(csv.reader(file))
    return data

def extract_movie_ratings(data):
    movie_ratings = {}
    for row in data[1:]:
        title = row[0]
        rating = float(row[2])
        if title in movie_ratings:
            movie_ratings[title].append(rating)
        else:
            movie_ratings[title] = [rating]
    return movie_ratings

def get_average_ratings(movies_dict):
    avg_ratings = {}
    for movie in movies_dict:
        ratings = movies_dict[movie]
        avg = sum(ratings) / len(ratings)
        avg_ratings[movie] = avg
    return avg_ratings

def print_top_movies(avg_ratings):
    sorted_movies = sorted(avg_ratings.items(), key=lambda x: x[1], reverse=True)
    for movie, rating in sorted_movies[:5]:
        print(f"{movie}: {rating:.2f}")

def main():
    file_path = 'movies.csv'
    data = read_file(file_path)
    movie_ratings = extract_movie_ratings(data)
    average_ratings = get_average_ratings(movie_ratings)
    print_top_movies(average_ratings)

main()
