from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

texts=[
    "win a free iphone now",
    "limited offer claim your prize",
    "meeting at 5 pm tomorrow",
    "can we have lunch today",
    "earn money fast click here",
    "this is not spam just a reminder",
]

labels= [
    "spam",
    "spam",
    "ham",
    "ham",
    "spam",
    "ham",
]

vectorizer=CountVectorizer()
X=vectorizer.fit_transform(texts)

model=MultinomialNB()
model.fit(X,labels)

while True:
    msg=input("Type a message (or 'q' to quit): ")
    if(msg.lower()=='q'):
        break

    X_new=vectorizer.transform([msg])
    pred=model.predict(X_new)[0]
    print("Prediction:", pred)