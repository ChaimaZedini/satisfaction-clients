# train.py - Entraînement et sauvegarde des modèles
import pandas as pd
import numpy as np
import re
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

print("=" * 60)
print("🎯 ENTRAÎNEMENT DU MODÈLE AVEC WORD2VEC")
print("=" * 60)

# 1. Télécharger NLTK
print("\n📥 Téléchargement des ressources NLTK...")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 2. Charger les données
print("\n📊 Chargement des données...")
try:
    df = pd.read_csv("C:/Users/GIGABYTE/Desktop/projet NLP/amazon_review.csv")
    print(f"✅ Fichier chargé : {df.shape[0]} avis")
except FileNotFoundError:
    print("❌ ERREUR : Fichier 'amazon_review.csv' non trouvé")
    exit()

# 3. Préparer les données
df = df[['overall', 'reviewText']].copy()

# Classes binaires
def convertir_note(note):
    return 1 if note >= 4 else 0

df['classe'] = df['overall'].apply(convertir_note)
print(f"📈 Distribution : Satisfait={sum(df['classe']==1)}, Non satisfait={sum(df['classe']==0)}")

# 4. Nettoyage du texte
print("\n🧹 Nettoyage du texte...")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def nettoyer_texte(texte):
    if pd.isna(texte):
        return []
    
    texte = str(texte).lower()
    texte = re.sub(r'[^a-z\s]', ' ', texte)
    
    mots = word_tokenize(texte)
    mots = [m for m in mots if m not in stop_words and len(m) > 2]
    mots = [lemmatizer.lemmatize(m) for m in mots]
    
    return mots

df['tokens'] = df['reviewText'].apply(nettoyer_texte)
df = df[df['tokens'].apply(len) > 0]
print(f"✅ Avis valides : {len(df)}")

# 5. Entraîner Word2Vec
print("\n🤖 Entraînement Word2Vec...")
word2vec_model = Word2Vec(
    sentences=df['tokens'].tolist(),
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    epochs=20,
    sg=1,
    seed=42
)
print(f"✅ Vocabulaire : {len(word2vec_model.wv)} mots")

# 6. Créer vecteurs documents
def document_vector(tokens, model):
    mots = [m for m in tokens if m in model.wv]
    if len(mots) == 0:
        return np.zeros(model.vector_size)
    return np.mean([model.wv[m] for m in mots], axis=0)

X = np.array([document_vector(t, word2vec_model) for t in df['tokens']])
y = df['classe'].values

print(f"🔧 Vecteurs créés : {X.shape}")

# 7. Entraîner classifieur
print("\n🎓 Entraînement classifieur...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(
    class_weight='balanced',
    random_state=42,
    max_iter=1000
)
model.fit(X_train, y_train)

# 8. Évaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"📈 Accuracy : {accuracy:.2%}")

# 9. Sauvegarde
print("\n💾 Sauvegarde des modèles...")

# Word2Vec
with open('word2vec_model.pkl', 'wb') as f:
    pickle.dump(word2vec_model, f)
print("✅ word2vec_model.pkl")

# Classifieur
with open('classifier_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ classifier_model.pkl")

# Statistiques
stats = {
    'accuracy': accuracy,
    'vocab_size': len(word2vec_model.wv),
    'dataset_size': len(df),
    'class_distribution': df['classe'].value_counts().to_dict()
}

with open('model_stats.pkl', 'wb') as f:
    pickle.dump(stats, f)
print("✅ model_stats.pkl")

print("\n" + "=" * 60)
print("🚀 MODÈLES PRÊTS POUR STREAMLIT!")
