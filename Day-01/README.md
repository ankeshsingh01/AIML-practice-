# Day 1 - Spam Message Classifier

Today I built a small NLP model that tells whether a text message is spam or not.
I used TF-IDF to turn the messages into numbers (since ML models can't read raw text), then trained a Naive Bayes classifier on top of it. Pretty classic combo for text classification, and it's the same idea behind a lot of real-world spam filters.
For testing, I used Leave-One-Out Cross Validation (LOOCV) instead of a normal train/test split. My dataset is tiny (only 16 messages), so a regular split felt kind of random - depending on which few messages ended up in the test set, accuracy could swing a lot. LOOCV tests the model 16 times, each time leaving out just one message, so the final number is way more trustworthy.
Honestly, the accuracy came out lower than I expected - 37.5%. First I thought something was broken, but it actually makes sense: 16 messages just isn't enough data for the model to learn real spam patterns. It's not a bug, it's a good reminder of why dataset size matters so much in ML. Kind of a useful thing to have actually seen happen instead of just reading about it.

### Stack
Python, Scikit-learn, TF-IDF, Naive Bayes, LOOCV

### Running it
```bash
pip install scikit-learn numpy
python spam_classifier.py
```

It'll print all 16 LOOCV rounds, the overall accuracy, and then test the model on two new messages I made up myself.

### What I'd do next
- Try this on a real dataset like the SMS Spam Collection (thousands of messages instead of 16) and see if accuracy actually goes up
- Compare Naive Bayes against Logistic Regression or SVM
- Maybe wrap it in a simple Flask app so I can type a message in and get a live prediction
