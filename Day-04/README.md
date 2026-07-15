# Day 4 - Movie Recommendation Engine

Built a "if you liked X, you'll like Y" recommender today - the same kind of thing Netflix/Amazon show you, just a much smaller, simpler version.

This is called **content-based filtering** - it recommends movies based on the movie's OWN content (genre, description tags), not based on what other users rated highly. The other common approach is **collaborative filtering** (look at users with similar taste and recommend what they liked), but that needs a big ratings dataset with lots of users, which I don't have here. Content-based is the more realistic option for a small project like this.

### How it actually works

1. Each movie has a "tags" string - basically genre + short description keywords (e.g. "action scifi space battle war heroic")
2. Turned those tags into TF-IDF vectors - same exact technique from Day 1's spam classifier, just applied to movie tags instead of text messages. Kind of cool seeing the same tool work on a completely different problem.
3. Computed **cosine similarity** between every pair of movies. This measures the angle between two vectors rather than their raw distance - so it cares about whether two movies "point in the same direction" content-wise, regardless of how long the tag list is. Score goes from 0 (nothing alike) to 1 (basically identical).
4. Built a `recommend()` function that, given a movie title, looks up its row in the similarity matrix, sorts all other movies by similarity score, and returns the top few.

### What I noticed testing it

Most recommendations made sense - "Galaxy Warriors" (scifi/action) recommended "Space Rebels" and "The Last Stand", both scifi/war movies. "Love in Paris" recommended other romance movies. Worked pretty much as expected.

One interesting edge case: when I asked for recommendations for "Detective Shadows" (a mystery/crime movie), it only had 2 genuinely similar movies in this tiny dataset. Since I asked for top 3, the third result was some random unrelated movie with a similarity score of literally 0.0 - meaning it has ZERO tags in common. That's not a bug, it's just what happens when you ask for more recommendations than the dataset can honestly support. In a real system you'd either have way more movies to draw from, or you'd set a minimum similarity threshold and just show fewer results if nothing else clears the bar.

### Stack
Python, Pandas, Scikit-learn (TfidfVectorizer, cosine_similarity)

### Running it
```bash
pip install pandas scikit-learn numpy
python movie_recommender.py
```

Tries 4 example movies and prints the top 3 recommendations for each, along with similarity scores.

### What I'd try next
- Add a minimum similarity threshold so it doesn't recommend near-zero matches just to fill the quota
- Use real movie data (like the MovieLens or TMDB dataset) with actual plot summaries instead of made-up tags
- Try collaborative filtering too, if I can find a dataset with user ratings, and compare the two approaches
