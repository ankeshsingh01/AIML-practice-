"""
Day 8 - DNA Sequence Classification (Gene Family Prediction)
------------------------------------------------------------------
Directly connects to my patent work (genomic DNA sequence analysis for
rice phenotype prediction) - same core idea, applied to human gene data
this time: given a raw DNA sequence (a string of A/T/G/C letters), predict
which of 7 gene families it belongs to.

Dataset: 4,380 real human DNA sequences, labeled with their gene family
(0-6). This is a well-known bioinformatics benchmark dataset.
Source: https://github.com/krishnaik06/DNA-Sequencing-Classifier

Concepts covered:
- K-mer counting - turning a DNA sequence into "words" a text-classification
  model can understand (a technique straight from Day 1's NLP work, applied
  to biology instead of language)
- Multi-class classification (7 classes, not just 2)
- Working with real genomic/biological sequence data
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# ---------------------------
# 1. Load the real dataset
# ---------------------------
df = pd.read_csv("human_data.txt", sep="\t")
print(f"Dataset: {len(df)} real human DNA sequences")
print(f"Gene family classes: {sorted(df['class'].unique())}")
print(df['class'].value_counts().sort_index())

# ---------------------------
# 2. Turn each DNA sequence into "k-mer words"
# A DNA sequence is just a long string like "ATGCCCCAACTAAAT..." - there are
# no natural "words" in it like in English text. The trick (borrowed
# straight from NLP): break the sequence into overlapping chunks of length
# k (called k-mers). E.g. with k=6, "ATGCCC" becomes one k-mer, then slide
# over by 1 letter: "TGCCCC", etc. Each k-mer becomes a "word" that
# CountVectorizer can count, exactly like it would count words in a sentence.
# ---------------------------
def get_kmers(sequence, k=6):
    return [sequence[i:i+k].lower() for i in range(len(sequence) - k + 1)]

df['words'] = df['sequence'].apply(lambda seq: get_kmers(seq, k=6))
df['text_form'] = df['words'].apply(lambda words: ' '.join(words))

print(f"\nExample - first sequence turned into k-mer 'sentence':")
print(df['text_form'].iloc[0][:150] + "...")

# ---------------------------
# 3. Vectorize the k-mer "sentences" using CountVectorizer (bag-of-words)
# Using 4-mers to 6-mers together (ngram_range) captures a bit more context
# ---------------------------
vectorizer = CountVectorizer(ngram_range=(4, 4))
X = vectorizer.fit_transform(df['text_form'])
y = df['class']

print(f"\nVocabulary size (unique k-mer combinations found): {len(vectorizer.vocabulary_)}")

# ---------------------------
# 4. Train/test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------
# 5. Train Naive Bayes (same algorithm family as Day 1's spam classifier -
# works well for this kind of count-based text/sequence data)
# ---------------------------
model = MultinomialNB(alpha=0.1)
model.fit(X_train, y_train)

# ---------------------------
# 6. Evaluate
# ---------------------------
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\nTest accuracy: {round(accuracy * 100, 2)}%")
print("\n", classification_report(y_test, predictions))

# ---------------------------
# 7. Confusion matrix across all 7 gene families
# ---------------------------
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(7, 6))
disp.plot(ax=ax, cmap="Greens", colorbar=False)
plt.title("Confusion Matrix - Gene Family Classification (7 classes)")
plt.tight_layout()
plt.savefig("images/confusion_matrix.png")
print("Saved images/confusion_matrix.png")

# ---------------------------
# 8. Class distribution chart
# ---------------------------
plt.figure(figsize=(7, 5))
df['class'].value_counts().sort_index().plot(kind='bar', color='#2ecc71')
plt.xlabel("Gene Family Class")
plt.ylabel("Number of sequences")
plt.title("Dataset Class Distribution (real human DNA sequences)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("images/class_distribution.png")
print("Saved images/class_distribution.png")
