# app.py - Version SIMPLIFIÉE sans problèmes d'import
import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle

# Configuration de la page
st.set_page_config(page_title="Analyse de Sentiment", layout="centered")

# Titre
st.title("🔍 Analyse de Sentiment Amazon")
st.write("Entrez un commentaire pour analyser la satisfaction client")

# Charger les modèles
@st.cache_resource
def charger_modeles():
    try:
        # Essayer de charger les modèles
        with open('word2vec_model.pkl', 'rb') as f:
            word2vec_model = pickle.load(f)
        with open('classifier_model.pkl', 'rb') as f:
            classifier_model = pickle.load(f)
        return word2vec_model, classifier_model
    except FileNotFoundError:
        st.error("⚠️ Les modèles ne sont pas trouvés. Exécutez d'abord 'python train_model_simple.py'")
        return None, None
    except Exception as e:
        st.error(f"Erreur: {e}")
        return None, None

# Chargement
word2vec_model, classifier_model = charger_modeles()

# Fonction de nettoyage SIMPLIFIÉE (sans NLTK)
def nettoyer_texte_simple(texte):
    """Nettoyage basique sans NLTK"""
    if not texte or pd.isna(texte):
        return []
    
    # Convertir en minuscules
    texte = str(texte).lower()
    
    # Supprimer les caractères spéciaux et chiffres
    texte = re.sub(r'[^a-z\s]', ' ', texte)
    
    # Séparer en mots
    mots = texte.split()
    
    # Liste de stopwords manuelle (basique)
    stopwords_simples = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
        'can', 'could', 'may', 'might', 'must', 'i', 'you', 'he', 'she', 'it',
        'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
        'its', 'our', 'their', 'this', 'that', 'these', 'those'
    }
    
    # Filtrer les mots courts et stopwords
    mots_filtres = [m for m in mots if len(m) > 2 and m not in stopwords_simples]
    
    return mots_filtres

def predire_sentiment(commentaire):
    """Prédire le sentiment"""
    if word2vec_model is None or classifier_model is None:
        return "Modèles non chargés", 0.0
    
    # Nettoyer le texte
    tokens = nettoyer_texte_simple(commentaire)
    
    if len(tokens) == 0:
        return "Texte trop court", 0.0
    
    # Créer vecteur document (simplifié)
    try:
        # Filtrer les mots dans le vocabulaire
        mots_valides = [m for m in tokens if m in word2vec_model.wv]
        
        if len(mots_valides) == 0:
            return "Texte non analysable", 0.0
        
        # Moyenne des vecteurs
        vecteurs = [word2vec_model.wv[m] for m in mots_valides]
        vecteur = np.mean(vecteurs, axis=0)
        
        # Prédiction
        prediction = classifier_model.predict([vecteur])[0]
        proba = classifier_model.predict_proba([vecteur])[0]
        
        if prediction == 1:
            return "✅ Client SATISFAIT", proba[1]
        else:
            return "❌ Client NON SATISFAIT", proba[0]
    
    except Exception as e:
        return f"Erreur d'analyse: {str(e)}", 0.0

# --- INTERFACE ---
# Zone de texte
commentaire = st.text_area(
    "Commentaire du client:",
    placeholder="Ex: 'The product is excellent! Very good quality.'",
    height=100
)

# Bouton
if st.button("🔍 Analyser", type="primary", use_container_width=True):
    if commentaire.strip():
        with st.spinner("Analyse en cours..."):
            resultat, confiance = predire_sentiment(commentaire)
        
        # Affichage résultat
        st.markdown("---")
        
        if "SATISFAIT" in resultat:
            st.success(f"## {resultat}")
        elif "NON SATISFAIT" in resultat:
            st.error(f"## {resultat}")
        else:
            st.warning(f"## {resultat}")
        
        if confiance > 0:
            st.metric("Confiance", f"{confiance:.1%}")
            st.progress(float(confiance))
    else:
        st.warning("Veuillez entrer un commentaire.")

# Exemples
st.markdown("---")
st.write("💡 **Exemples à tester:**")

exemples = [
    "This product is amazing! Perfect quality.",
    "Terrible experience, never buying again.",
    "Good product but delivery was late.",
    "Absolutely love it! 5 stars.",
    "Waste of money, very disappointed."
]

# Afficher les exemples comme des boutons
for i, exemple in enumerate(exemples):
    if st.button(f"Exemple {i+1}: {exemple[:30]}...", key=f"ex_{i}"):
        # Met à jour la zone de texte
        st.session_state.text_input = exemple
        st.rerun()

# Pied de page
st.markdown("---")
st.caption("Analyse de sentiment - Projet Fouille de Données")