"""
Day 4 - Movie Recommendation Engine (Content-Based Filtering)
------------------------------------------------------------------
Given a movie the user likes, recommend similar movies based on
genre/description content - no user ratings needed, just the movies'
own features. This is called "content-based filtering" (as opposed to
"collaborative filtering", which looks at what OTHER users with similar
taste liked - that needs a big ratings dataset, which I don't have here).

Concepts covered:
- TF-IDF on movie descriptions/genres (same technique as Day 1, new use case)
- Cosine similarity - measuring how "close" two vectors point in the same direction
- Building a simple recommendation function on top of a similarity matrix
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# 1. Small built-in movie dataset
# "tags" = combined genres + short description, this is what we compare movies on
# ---------------------------
movies = pd.DataFrame({
    "title": [
        "Galaxy Warriors", "Space Rebels", "The Last Stand",
        "Love in Paris", "Midnight Romance", "Heartbreak Hotel",
        "Detective Shadows", "The Silent Witness", "Crime City",
        "Laugh Out Loud", "Comedy Night", "The Funny Guy",
        "Alien Invasion", "Starship Odyssey", "Robot Uprising",
        "Wedding Bells", "Second Chances",
    ],
    "tags": [
        "action scifi space battle war heroic",
        "scifi space rebellion war action battle",
        "action war heroic last battle survival",
        "romance drama paris love story emotional",
        "romance drama love story emotional night",
        "romance drama heartbreak love emotional",
        "mystery crime detective thriller investigation",
        "mystery crime thriller silent investigation witness",
        "crime action city thriller investigation",
        "comedy funny laugh humor lighthearted",
        "comedy funny night humor lighthearted",
        "comedy funny guy humor lighthearted",
        "scifi action alien invasion battle war",
        "scifi space odyssey battle exploration",
        "scifi action robot uprising battle war",
        "romance comedy wedding love lighthearted",
        "romance drama second chances emotional love",
    ]
})

# ---------------------------
# 2. Convert the "tags" text into TF-IDF vectors
# Same technique as the Day 1 spam classifier, just applied to movie tags instead of messages
# ---------------------------
vectorizer = TfidfVectorizer(stop_words="english")
tag_vectors = vectorizer.fit_transform(movies["tags"])

# ---------------------------
# 3. Compute cosine similarity between every pair of movies
# Cosine similarity measures the angle between two vectors, ignoring their length -
# it answers "do these two movies point in the same direction content-wise?"
# Score range: 0 (nothing in common) to 1 (identical content)
# ---------------------------
similarity_matrix = cosine_similarity(tag_vectors)

# ---------------------------
# 4. Recommendation function
# ---------------------------
def recommend(movie_title, top_n=3):
    if movie_title not in movies["title"].values:
        print(f"'{movie_title}' not found in the dataset.")
        return

    movie_index = movies[movies["title"] == movie_title].index[0]
    similarity_scores = list(enumerate(similarity_matrix[movie_index]))

    # sort by similarity score, highest first, and skip the movie itself (always similarity=1 to itself)
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = [s for s in similarity_scores if s[0] != movie_index][:top_n]

    print(f"\nBecause you liked '{movie_title}', you might also like:")
    for index, score in similarity_scores:
        print(f"  - {movies.iloc[index]['title']}  (similarity: {round(score, 3)})")


# ---------------------------
# 5. Try it out on a few movies
# ---------------------------
recommend("Galaxy Warriors")
recommend("Love in Paris")
recommend("Detective Shadows")
recommend("Laugh Out Loud")
