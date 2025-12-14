from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os

app = Flask(__name__)

# Check if model files exist, if not - train and save
model_file = 'spam_model.pkl'
vectorizer_file = 'vectorizer.pkl'

if os.path.exists(model_file) and os.path.exists(vectorizer_file):
    # Load saved model/vectorizer
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_file, 'rb') as f:
        vectorizer = pickle.load(f)
    print("Loaded saved model!")
else:
    # Train new model (first run only)
    print("Training new model...")
    texts = [
        "win a free iphone now", "limited offer claim your prize", "meeting at 5 pm tomorrow",
        "can we have lunch today", "earn money fast click here", "this is not spam just a reminder"
    ]
    labels = ["spam", "spam", "ham", "ham", "spam", "ham"]
    
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    model = MultinomialNB()
    model.fit(X, labels)
    
    # Save model and vectorizer
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    with open(vectorizer_file, 'wb') as f:
        pickle.dump(vectorizer, f)
    print("Model saved!")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ""
    if request.method == 'POST':
        message = request.form.get('message', '')
        X_new = vectorizer.transform([message])
        prediction = model.predict(X_new)[0]
    
    return render_template('spam.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
