import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Télécharger les ressources NLTK (caché à l'utilisateur)
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except:
    nltk.download('wordnet', quiet=True)

# ============================================
# CONFIGURATION DE L'INTERFACE
# ============================================

st.set_page_config(
    page_title="Analyse de Sentiment Amazon",
    page_icon="😊",
    layout="centered"
)

# Titre
st.title("😊 Analyse de Satisfaction Client")
st.markdown("---")

# ============================================
# CHARGEMENT DES MODÈLES
# ============================================

@st.cache_resource
def charger_modeles():
    """Charger les modèles Word2Vec et le classifieur"""
    try:
        # Charger Word2Vec
        with open('word2vec_model.pkl', 'rb') as f:
            word2vec_model = pickle.load(f)
        
        # Charger le classifieur
        with open('classifier_model.pkl', 'rb') as f:
            classifier_model = pickle.load(f)
        
        # Charger les statistiques
        with open('model_stats.pkl', 'rb') as f:
            stats = pickle.load(f)
        
        return word2vec_model, classifier_model, stats
    
    except FileNotFoundError:
        st.error("""
        ⚠️ **Modèles non trouvés !**
        
        Suivez ces étapes :
        1. Exécutez d'abord le notebook pour entraîner les modèles
        2. Assurez-vous que ces fichiers sont présents :
           - `word2vec_model.pkl`
           - `classifier_model.pkl`
           - `model_stats.pkl`
        """)
        return None, None, None
    
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        return None, None, None

# Charger les modèles
word2vec_model, classifier_model, stats = charger_modeles()

# ============================================
# FONCTIONS DE TRAITEMENT
# ============================================

# Initialiser les outils NLP
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def nettoyer_et_tokeniser(texte):
    """Nettoyer et tokeniser un texte"""
    if not texte or pd.isna(texte):
        return []
    
    texte = str(texte).lower()
    
    # Supprimer caractères spéciaux
    texte = re.sub(r'[^a-z\s]', ' ', texte)
    
    # Tokenisation simple
    mots = texte.split()
    
    # Supprimer stopwords et mots courts
    mots = [m for m in mots if m not in stop_words and len(m) > 2]
    
    # Lemmatisation
    mots = [lemmatizer.lemmatize(m) for m in mots]
    
    return mots

def creer_vecteur_document(tokens, model):
    """Créer un vecteur document à partir des tokens"""
    if not tokens:
        return np.zeros(100)  # 100 = dimension Word2Vec
    
    # Filtrer les mots dans le vocabulaire
    mots_valides = [m for m in tokens if m in model.wv]
    
    if not mots_valides:
        return np.zeros(100)
    
    # Moyenne des vecteurs
    return np.mean([model.wv[m] for m in mots_valides], axis=0)

def predire_sentiment(commentaire):
    """Fonction principale de prédiction"""
    if word2vec_model is None or classifier_model is None:
        return "Modèles non chargés", 0.0, []
    
    # 1. Nettoyer le texte
    tokens = nettoyer_et_tokeniser(commentaire)
    
    # 2. Vérifier si valide
    if len(tokens) == 0:
        return "Texte trop court", 0.0, []
    
    # 3. Créer vecteur document
    vecteur_doc = creer_vecteur_document(tokens, word2vec_model)
    
    # 4. Prédiction
    prediction = classifier_model.predict([vecteur_doc])[0]
    proba = classifier_model.predict_proba([vecteur_doc])[0]
    confiance = proba[prediction]
    
    # 5. Résultat
    if prediction == 1:
        sentiment = "✅ Client SATISFAIT"
    else:
        sentiment = "❌ Client NON SATISFAIT"
    
    return sentiment, confiance, tokens

# ============================================
# INTERFACE UTILISATEUR
# ============================================

# Barre latérale avec informations
st.sidebar.header("📊 Informations du modèle")
if stats:
    st.sidebar.metric("Accuracy", f"{stats.get('accuracy', 0):.2%}")
    st.sidebar.metric("Taille du vocabulaire", f"{stats.get('vocab_size', 0):,}")
    if 'class_distribution' in stats:
        dist = stats['class_distribution']
        st.sidebar.write("**Répartition :**")
        st.sidebar.write(f"- Satisfait : {dist.get(1, 0)}")
        st.sidebar.write(f"- Non satisfait : {dist.get(0, 0)}")

# Section principale
st.subheader("📝 Analysez un commentaire")

# Zone de texte
commentaire = st.text_area(
    "Entrez votre commentaire ci-dessous :",
    height=120,
    placeholder="Exemple : 'This product is excellent! Very good quality and fast delivery.'",
    key="input_text"
)

# Bouton d'analyse
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("🔍 Analyser", type="primary", use_container_width=True):
        st.session_state.analyser = True

# Si l'utilisateur clique sur Analyser
if hasattr(st.session_state, 'analyser') and st.session_state.analyser:
    if commentaire.strip():
        with st.spinner("Analyse en cours avec Word2Vec..."):
            # Prédiction
            sentiment, confiance, tokens = predire_sentiment(commentaire)
        
        # Affichage des résultats
        st.markdown("---")
        st.subheader("📊 Résultat de l'analyse")
        
        # Affichage du sentiment
        if "SATISFAIT" in sentiment:
            st.success(f"## {sentiment}")
            st.balloons()
        else:
            st.error(f"## {sentiment}")
        
        # Métriques
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Confiance", f"{confiance:.1%}")
        with col_b:
            st.metric("Mots analysés", len(tokens))
        
        # Barre de progression
        st.progress(float(confiance))
        
        # Détails
        with st.expander("🔍 Voir les détails d'analyse"):
            st.write("**Texte analysé :**")
            st.info(f'"{commentaire}"')
            
            st.write("**Mots extraits :**")
            if tokens:
                # Afficher les mots avec des badges colorés
                html_tokens = ""
                for token in tokens[:30]:  # Limiter à 30 mots
                    if token in word2vec_model.wv:
                        html_tokens += f'<span style="background:#4CAF50;color:white;padding:3px 8px;margin:2px;border-radius:5px;display:inline-block;">{token}</span> '
                    else:
                        html_tokens += f'<span style="background:#ff9800;color:white;padding:3px 8px;margin:2px;border-radius:5px;display:inline-block;">{token}</span> '
                
                st.markdown(html_tokens, unsafe_allow_html=True)
                
                if len(tokens) > 30:
                    st.write(f"... et {len(tokens) - 30} autres mots")
            else:
                st.write("Aucun mot extrait")
            
            # Information technique
            if word2vec_model:
                st.write(f"**Dimension Word2Vec :** {word2vec_model.vector_size}")
    
    else:
        st.warning("⚠️ Veuillez entrer un commentaire avant d'analyser.")

# Section exemples
st.markdown("---")
st.subheader("💡 Exemples rapides à tester")

exemples = [
    "This product is amazing! Perfect quality and fast delivery.",
    "Terrible experience. The item broke after 2 days of use.",
    "Good value for money but shipping was a bit slow.",
    "Absolutely love it! Best purchase I've made this year.",
    "Waste of money. Very disappointed with the quality."
]

# Afficher les boutons d'exemples
cols = st.columns(5)
for i, exemple in enumerate(exemples):
    with cols[i]:
        if st.button(f"Ex {i+1}", key=f"btn_{i}"):
            # Mettre à jour la zone de texte
            st.session_state.input_text = exemple
            # Déclencher une nouvelle exécution
            st.rerun()

# Instructions d'utilisation
st.markdown("---")
with st.expander("ℹ️ Comment utiliser cette application"):
    st.write("""
    **Instructions :**
    1. Écrivez un commentaire en anglais dans la zone de texte
    2. Cliquez sur le bouton **"Analyser"**
    3. Consultez le résultat de l'analyse
    
    **Technologie utilisée :**
    - **Word2Vec** : Modèle d'embedding de mots
    - **Régression Logistique** : Classifieur binaire
    
    **Classes de résultat :**
    - ✅ **Client SATISFAIT** : Correspond aux notes 4-5 étoiles
    - ❌ **Client NON SATISFAIT** : Correspond aux notes 1-3 étoiles
    
    **Note :** Pour de meilleurs résultats, utilisez des commentaires en anglais.
    """)

# Pied de page
st.markdown("---")
st.caption("Projet de Fouille de Données - Analyse de Sentiment avec Word2Vec")