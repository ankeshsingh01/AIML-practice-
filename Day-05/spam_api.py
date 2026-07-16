"""
Day 5 - Deploying the Spam Classifier as a FastAPI Service
------------------------------------------------------------
Everything so far (Day 1-4) was a script - run it, see printed output, done.
That's fine for learning, but it's not how ML actually gets used in the real
world. In practice, a model needs to sit behind an API so OTHER programs
(a website, a mobile app, another service) can send it data and get a
prediction back, on demand.

Today I took the Day 1 spam classifier and wrapped it in a FastAPI app with
a single endpoint: send it a message, get back whether it's spam.

Concepts covered:
- What an API actually is, in practice (a program that answers other programs, not humans, directly)
- FastAPI basics - defining a POST endpoint, request/response models with Pydantic
- Training a model once at startup vs. retraining on every request (huge deal for real latency)
- Testing an API with the built-in interactive docs (Swagger UI)
"""

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ---------------------------
# 1. Same small dataset from Day 1
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
# 2. Train the model ONCE, when the server starts up
# This is the key difference from a script: an API stays running and handles
# many requests, so you train once and reuse the same trained model for
# every incoming request instead of retraining every time (which would be
# painfully slow if this were a bigger, real model).
# ---------------------------
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(messages)
model = MultinomialNB()
model.fit(X, labels)

# ---------------------------
# 3. Define the API
# ---------------------------
app = FastAPI(title="Spam Classifier API")

# Pydantic model - this defines exactly what shape of JSON the API expects
# in the request body. FastAPI uses this to auto-validate incoming requests
# and to generate the interactive docs.
class MessageRequest(BaseModel):
    message: str


class PredictionResponse(BaseModel):
    message: str
    prediction: str
    confidence: float


@app.get("/")
def home():
    return {"status": "Spam Classifier API is running. Go to /docs to try it out."}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: MessageRequest):
    message_vector = vectorizer.transform([request.message])
    prediction = model.predict(message_vector)[0]
    # predict_proba gives the probability for each class - taking the max
    # as a rough "confidence" score
    probabilities = model.predict_proba(message_vector)[0]
    confidence = round(max(probabilities), 3)

    return PredictionResponse(
        message=request.message,
        prediction=prediction,
        confidence=confidence
    )


# ---------------------------
# To run this:
#   pip install fastapi uvicorn scikit-learn
#   uvicorn spam_api:app --reload
# Then open http://127.0.0.1:8000/docs in a browser to test it interactively
# ---------------------------
