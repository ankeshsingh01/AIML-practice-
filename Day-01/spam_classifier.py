
What this does:
1. Takes a small dataset of text messages labeled as "spam" or "ham" (not spam)
2. Converts text into numbers using TF-IDF (Term Frequency - Inverse Document Frequency)
3. Trains a Naive Bayes classifier to learn the difference between spam and normal messages
4. Tests the model on new, unseen messages

Concepts learned today:
- Text preprocessing
- TF-IDF vectorization
- Leave-One-Out Cross Validation (LOOCV)
- Naive Bayes classification
- Model evaluation (accuracy)

Why LOOCV instead of a simple train/test split?
With only a small number of samples, a single train/test split can be misleading
(the result depends heavily on which few messages ended up in the test set).
LOOCV instead trains the model N times -- each time leaving exactly ONE message
out as the test sample and training on all the rest -- then averages the results.
This gives a much more reliable accuracy estimate for small datasets.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import LeaveOneOut
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------
# 1. Sample dataset (small, built-in so no download needed)
# ---------------------------
messages = [
    "Congratulations! You won a free iPhone, click here to claim now",
    "Hey, are we still meeting for lunch tomorrow?",
    "URGENT: Your account has been suspended, verify immediately",
    "Can you send me the notes from today's class?",
    "You have been selected for a cash prize of $1000, reply now",
    "Don't forget to bring your laptop for the meeting",
    "FREE entry in a weekly competition to win an iPad, text WIN to 80086",
    "Mom, I'll be home by 8pm, saving you some dinner",
    "Claim your free gift card now, limited time offer",
    "Let's catch up this weekend, it's been a while",
    "You have won a lottery of Rs 50,00,000! Send your bank details",
    "Please review the attached document before our call",
    "Exclusive deal just for you, click the link to save 90% today",
    "Happy birthday! Hope you have a wonderful day",
    "Get rich quick, invest now and double your money in a week",
    "Reminder: your dentist appointment is at 4pm today",
]

labels = [
    "spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham",
    "spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham"
]

# ---------------------------
# 2. Set up Leave-One-Out Cross Validation
# ---------------------------
messages = np.array(messages)
labels = np.array(labels)
loo = LeaveOneOut()

y_true_all = []
y_pred_all = []

# ---------------------------
# 3 & 4. For each split: vectorize with TF-IDF, train Naive Bayes, predict the one held-out message
# ---------------------------
for train_index, test_index in loo.split(messages):
    X_train, X_test = messages[train_index], messages[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    vectorizer = TfidfVectorizer(stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    prediction = model.predict(X_test_vec)
    y_true_all.append(y_test[0])
    y_pred_all.append(prediction[0])

# ---------------------------
# 5. Evaluate the model across all LOOCV rounds
# ---------------------------
print(f"Total LOOCV rounds: {len(messages)} (one per message)\n")
print("Actual Labels:   ", y_true_all)
print("Predicted Labels:", y_pred_all)
print("\nOverall LOOCV Accuracy:", accuracy_score(y_true_all, y_pred_all))
print("\nDetailed Report:\n", classification_report(y_true_all, y_pred_all, zero_division=0))

# Refit the vectorizer + model on the FULL dataset for the custom test below
vectorizer = TfidfVectorizer(stop_words="english")
X_all_vec = vectorizer.fit_transform(messages)
model = MultinomialNB()
model.fit(X_all_vec, labels)

# ---------------------------
# 6. Try it on your own custom message
# ---------------------------
custom_messages = [
    "Win a brand new car by clicking this link now",
    "Are you free to talk tonight?",
]
custom_vec = vectorizer.transform(custom_messages)
custom_preds = model.predict(custom_vec)

print("\n--- Custom Test ---")
for msg, pred in zip(custom_messages, custom_preds):
    print(f"Message: '{msg}' --> Predicted: {pred}")
