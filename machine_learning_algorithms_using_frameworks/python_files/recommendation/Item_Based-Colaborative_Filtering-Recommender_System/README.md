# Item Based Colaborative Filtering  for Recommendation
Go through this [Report](https://docs.google.com/document/d/1_HuEZYfmOBlCWAlKkjJDSJCP7Ccf1UAYeFLHAHG9omo/edit?usp=sharing) to get details of  concepts used.

**Input**

    *   Given the ratings file(ratings.csv) of  1,000,209 anonymous ratings of approximately 3,900 movies made by 6,040 MovieLens users who joined MovieLens in 2000(Movielens           1M data set).
    *   Column Format : - UserID::MovieID::Rating::Timestamp
    *   UserIDs range between 1 and 6040 
    *   MovieIDs range between 1 and 3952
    *   Ratings are made on a 5-star scale (whole-star ratings only)
    *   Timestamp is represented in seconds since the epoch as returned by time(2)
    *   Each user has at least 20 ratings

**OutPut**

    Predict the ratings for a movie for the Users who didn’t watch that movie.

**Technical Description**

    *   Get the required data from here [data set](https://grouplens.org/datasets/movielens/1m/).
    *   File description 
        *   **ItemBasedCollaborativeFiltering.py** : - The code predicts the ratings based on Item similarity and uses Adjusted Cosine  for computing the similarity.  To Test                                                           the results data are divided into training and test Sets.
        *   **ItemBasedCollaborativeFilteringWith_5Fold.py : -** Here the data is divided into 5 fold to get accuracy of the system with training:test as 80:20 ratio and uses                                                                    the item based similarity for recommendations.
    *   Code can be executed either through any ide built in run module(i.e. spyder) or using command line , provided data should present beforehand.
    *   We set a Threshold after computing the rating of a Movie for Particular users as 4, If the predicted rating is below 4 , we don’t recommend that movie to users.
