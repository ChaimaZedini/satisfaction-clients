# train_model.py - Entraîner les modèles avant d'utiliser l'app
import pandas as pd
import numpy as np
import re
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

print(" Entraînement du modèle...")

# Télécharger NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Charger données
df = pd.read_csv("C:/Users/GIGABYTE/Desktop/projet NLP/amazon_review.csv")
df = df[['overall', 'reviewText']].copy()

# Classes binaires
df["classe"] = df["overall"].apply(lambda x: 1 if x >= 4 else 0)

# Nettoyage
def nettoyer(texte):
    if pd.isna(texte):
        return []
    texte = str(texte).lower()
    texte = re.sub(r'[^a-z\s]', ' ', texte)
    mots = word_tokenize(texte)
    stop_words = set(stopwords.words('english'))
    mots = [m for m in mots if m not in stop_words and len(m) > 2]
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(m) for m in mots]

df["tokens"] = df["reviewText"].apply(nettoyer)
df = df[df["tokens"].apply(len) > 0]

# Word2Vec
print(" Entraînement Word2Vec...")
word2vec_model = Word2Vec(
    sentences=df["tokens"].tolist(),
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    epochs=20
)

# Vecteurs documents
def doc_vector(tokens):
    mots = [m for m in tokens if m in word2vec_model.wv]
    if len(mots) == 0:
        return np.zeros(100)
    return np.mean([word2vec_model.wv[m] for m in mots], axis=0)

X = np.array([doc_vector(tokens) for tokens in df["tokens"]])
y = df["classe"].values

# Entraînement classifier
print("🎓 Entraînement classifieur...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)

# Évaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f" Accuracy: {accuracy:.2%}")

# Sauvegarde
with open('word2vec_model.pkl', 'wb') as f:
    pickle.dump(word2vec_model, f)

with open('classifier_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print(" Modèles sauvegardés: word2vec_model.pkl & classifier_model.pkl")
