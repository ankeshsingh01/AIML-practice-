# Day 5 - Deploying the Spam Classifier as an API

Everything from Day 1 to Day 4 was a script - run it, read the printed output in the terminal, done. Today's different: I took the Day 1 spam classifier and turned it into an actual API using FastAPI, so any other program (a website, a mobile app, anything) can send it a message over the internet and get a prediction back.

This is basically the difference between "I built a model" and "I shipped a model" - and it's the part a lot of tutorials skip.

### How it works

- Same TF-IDF + Naive Bayes setup from Day 1, but instead of training then immediately testing in one script run, the model trains ONCE when the server starts up and then just sits in memory, ready to answer requests. This matters a lot in practice - if this were a bigger real model, retraining it on every single request would be way too slow. Train once, reuse forever (until you decide to retrain with new data).
- Built one endpoint: `POST /predict` - send it a JSON body like `{"message": "your text here"}`, get back the prediction and a confidence score.
- Used Pydantic models (`MessageRequest`, `PredictionResponse`) to define exactly what shape of data goes in and comes out. FastAPI uses these to auto-validate requests (so if someone sends garbage data, it rejects it with a clear error instead of crashing) and to auto-generate interactive API docs.

### Testing it

FastAPI gives you a free interactive docs page at `/docs` (Swagger UI) - you can literally test the API from the browser without writing any client code. I also tested it with curl:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"message": "You won a free prize, click now!"}'
# -> {"message": "...", "prediction": "spam", "confidence": 0.719}

curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"message": "Are we still on for dinner tonight?"}'
# -> {"message": "...", "prediction": "ham", "confidence": 0.594}
```

Both came back correct.

### Stack
Python, FastAPI, Uvicorn, Pydantic, Scikit-learn

### Running it
```bash
pip install fastapi uvicorn scikit-learn
uvicorn spam_api:app --reload
```
Then open `http://127.0.0.1:8000/docs` in a browser to try it interactively, or hit the `/predict` endpoint directly with curl/Postman.

### What I'd try next
- Save the trained model to disk (with `pickle` or `joblib`) instead of retraining every time the server restarts
- Actually deploy this somewhere public (Render, Railway, or similar) instead of just running it on localhost
- Add basic error handling for empty/weird input
- Add a rate limiter so the API can't be spammed with requests (ironic, given it's a spam detector)
