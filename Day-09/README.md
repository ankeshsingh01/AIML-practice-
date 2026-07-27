# Day 8 - DNA Sequence Classification (Gene Family Prediction)

This one connects directly to my patent work (genomic DNA sequence analysis for rice phenotype prediction) - same core problem, applied here to human gene data: given a raw DNA sequence (just a string of A/T/G/C letters), predict which of 7 gene families it belongs to.

## The dataset

4,380 real human DNA sequences, each labeled with its gene family (classes 0-6). This is a well-known bioinformatics benchmark dataset.

Source: https://github.com/krishnaik06/DNA-Sequencing-Classifier

## The core trick: treating DNA like text

A DNA sequence is just a long string like `ATGCCCCAACTAAAT...` - there's no natural "words" in it the way English has words. So I borrowed a technique straight from NLP (same family of ideas as Day 1's spam classifier): break each sequence into overlapping **k-mers** - chunks of k letters, sliding one letter at a time. With k=6: `ATGCCC`, `TGCCCC`, `GCCCCA`, etc. Each k-mer becomes a "word," and then CountVectorizer counts them exactly like it would count words in a sentence. This turns raw biology into something a standard text-classification model can work with directly.

Used 4-mers specifically for the final vectorization (`ngram_range=(4,4)`) - short enough to catch common local patterns, but still specific enough to be meaningful. Ended up with 232,414 unique 4-mer combinations across the dataset - that's the "vocabulary" size, same concept as vocabulary size in a text classifier.

## Model and results

Trained Multinomial Naive Bayes (same algorithm as Day 1's spam classifier) on the k-mer counts.

**Test accuracy: 98.17%** across all 7 gene family classes - genuinely strong for a 7-way classification problem. Per-class performance was consistently high (precision/recall mostly 95-100%), even for the smallest class (class 5, only 240 samples total) which still hit 92% recall.

The dataset isn't perfectly balanced - class 6 has 1,343 sequences while class 5 has only 240 - and the confusion matrix (`images/confusion_matrix.png`) confirms the model handled this imbalance well without collapsing into just predicting the majority class.

## Stack
Python, Pandas, Scikit-learn (CountVectorizer, MultinomialNB), Matplotlib

## Running it
```bash
pip install pandas scikit-learn matplotlib
python dna_classifier.py
```
Needs `human_data.txt` in the same folder (the raw dataset file).

## What I'd try next
- Try different k-mer sizes (k=4 vs k=6 vs k=8) and see how accuracy changes - there's a real tradeoff between capturing enough context and vocabulary size exploding
- Try this same k-mer approach on my actual rice genotype data from the patent work, as a simpler baseline to compare against the CNN+Transformer architecture
- Look closer at which specific sequences in the smaller classes get misclassified
