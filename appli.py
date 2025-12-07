# appl.py
import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import joblib
import os

# ===========================================
# 1. CONFIGURATION DE LA PAGE
# ===========================================
st.set_page_config(
    page_title="Analyse de Sentiment",
    page_icon="😊",
    layout="centered"
)

# ===========================================
# 2. TITRE ET DESCRIPTION
# ===========================================
st.title("📊 Analyse de Sentiment Amazon")
st.markdown("**Satisfait ou Non Satisfait?** Découvrez-le en analysant votre commentaire.")
st.markdown("---")

# ===========================================
# 3. INITIALISATION NLTK
# ===========================================
@st.cache_resource
def setup_nltk():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        return True
    except:
        return False

if setup_nltk():
    st.sidebar.success("NLTK initialisé")
else:
    st.sidebar.warning("Problème NLTK")

# ===========================================
# 4. FONCTION DE NETTOYAGE
# ===========================================
def nettoyer_texte(texte):
    """Nettoie le texte pour l'analyse"""
    if not texte or pd.isna(texte):
        return ""
    
    # Convertir en minuscules
    texte = str(texte).lower()
    
    # Supprimer les caractères spéciaux et chiffres
    texte = re.sub(r'[^a-z\s]', ' ', texte)
    
    # Tokenisation
    mots = word_tokenize(texte)
    
    # Supprimer les stopwords
    stop_words = set(stopwords.words('english'))
    mots = [mot for mot in mots if mot not in stop_words and len(mot) > 2]
    
    # Lemmatisation
    lemmatizer = WordNetLemmatizer()
    mots = [lemmatizer.lemmatize(mot) for mot in mots]
    
    return " ".join(mots)

# ===========================================
# 5. CHARGEMENT DU MODÈLE (CORRIGÉ)
# ===========================================
@st.cache_resource
def charger_modele():
    """Charge le modèle et le vectorizer"""
    try:
        # Vérifie si les fichiers existent
        if os.path.exists('modele_sentiment.pkl') and os.path.exists('vectorizer.pkl'):
            model = joblib.load('modele_sentiment.pkl')
            vectorizer = joblib.load('vectorizer.pkl')
            return model, vectorizer, True
        else:
            st.warning("⚠️ Fichiers modèle non trouvés. Mode simulation activé.")
            return None, None, False
    except Exception as e:
        st.error(f"❌ Erreur de chargement: {str(e)}")
        return None, None, False

# Chargement
model, vectorizer, modele_pret = charger_modele()

# Message d'état
if modele_pret:
    st.sidebar.success("✅ Modèle chargé")
else:
    st.sidebar.warning("🔧 Mode simulation")

# ===========================================
# 6. INTERFACE UTILISATEUR
# ===========================================
st.subheader("✍️ Entrez votre commentaire")

# Zone de texte
commentaire = st.text_area(
    "",
    placeholder="Exemple: This product is amazing! The quality exceeded my expectations...",
    height=120,
    key="input_text"
)

# Boutons
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    btn_analyser = st.button("🔍 Analyser", type="primary", use_container_width=True)
with col2:
    btn_effacer = st.button("🧹 Effacer", use_container_width=True)

# Effacer le texte
if btn_effacer:
    st.rerun()

# ===========================================
# 7. ANALYSE DU COMMENTAIRE
# ===========================================
if btn_analyser:
    if not commentaire or commentaire.strip() == "":
        st.error("❌ Veuillez entrer un commentaire")
    else:
        with st.spinner("Analyse en cours..."):
            # Nettoyer le texte
            texte_propre = nettoyer_texte(commentaire)
            
            # ===========================================
            # CORRECTION PRINCIPALE : VÉRIFIER vectorizer
            # ===========================================
            if modele_pret and vectorizer is not None:
                # VÉRIFICATION IMPORTANTE
                if hasattr(vectorizer, 'transform'):
                    # Transformation avec TF-IDF
                    vect = vectorizer.transform([texte_propre])
                    
                    # Prédiction
                    prediction = model.predict(vect)[0]
                    probabilites = model.predict_proba(vect)[0]
                    
                    # Déterminer le sentiment
                    if prediction == '1':
                        sentiment = "✅ SATISFAIT"
                        confiance = probabilites[1] * 100
                        couleur = "green"
                    else:
                        sentiment = "❌ NON SATISFAIT"
                        confiance = probabilites[0] * 100
                        couleur = "red"
                else:
                    # Fallback si vectorizer invalide
                    st.warning("Problème avec le vectorizer, mode simulation")
                    modele_pret = False
            else:
                # Mode simulation
                modele_pret = False
            
            # ===========================================
            # MODE SIMULATION (si modèle non chargé)
            # ===========================================
            if not modele_pret:
                # Mots-clés pour la simulation
                mots_positifs = ['good', 'great', 'excellent', 'amazing', 'love', 
                               'perfect', 'happy', 'recommend', 'awesome', 'best']
                mots_negatifs = ['bad', 'terrible', 'poor', 'awful', 'hate', 
                               'disappointed', 'broken', 'worst', 'waste', 'avoid']
                
                # Compter les occurrences
                score_pos = sum(1 for mot in mots_positifs if mot in texte_propre)
                score_neg = sum(1 for mot in mots_negatifs if mot in texte_propre)
                
                # Décision
                if score_pos > score_neg:
                    sentiment = "✅ SATISFAIT"
                    confiance = min(80 + score_pos * 3, 95)
                    couleur = "green"
                elif score_neg > score_pos:
                    sentiment = "❌ NON SATISFAIT"
                    confiance = min(80 + score_neg * 3, 95)
                    couleur = "red"
                else:
                    sentiment = "🤷 NEUTRE"
                    confiance = 50
                    couleur = "orange"
            
            # ===========================================
            # 8. AFFICHAGE DES RÉSULTATS
            # ===========================================
            st.markdown("---")
            st.subheader("📊 Résultat")
            
            # Afficher le sentiment en grand
            st.markdown(f"<h1 style='text-align: center; color: {couleur};'>{sentiment}</h1>", 
                       unsafe_allow_html=True)
            
            # Barre de confiance
            st.markdown(f"**Confiance : {confiance:.1f}%**")
            st.progress(int(confiance) / 100)
            
            # Effets visuels
            if "SATISFAIT" in sentiment and couleur == "green":
                st.balloons()
            
            # Détails (optionnel)
            with st.expander("📝 Détails de l'analyse"):
                st.write("**Commentaire original :**")
                st.write(commentaire[:200] + "..." if len(commentaire) > 200 else commentaire)
                
                st.write("**Texte nettoyé :**")
                st.write(texte_propre[:200] + "..." if len(texte_propre) > 200 else texte_propre)
                
                if modele_pret:
                    st.write("**Source :** Modèle entraîné")
                else:
                    st.write("**Source :** Simulation (mots-clés)")

# ===========================================
# 9. INFORMATIONS
# ===========================================
st.markdown("---")
st.markdown("### ℹ️ À propos")
st.markdown("""
Cette application analyse le sentiment des commentaires Amazon :
- **✅ SATISFAIT** : Avis positif (note 4-5 étoiles)
- **❌ NON SATISFAIT** : Avis négatif (note 1-3 étoiles)

Le texte est automatiquement nettoyé (stopwords, lemmatisation).
""")

# Pied de page
st.markdown("---")
st.caption("Développé avec Streamlit | Projet d'analyse de sentiment")

# ===========================================
# 10. INSTRUCTIONS POUR LE MODÈLE
# ===========================================
with st.sidebar:
    st.markdown("### 🔧 Configuration")
    
    if st.button("Vérifier les fichiers modèle"):
        if os.path.exists('modele_sentiment.pkl'):
            st.success("modele_sentiment.pkl ✓")
        else:
            st.error("modele_sentiment.pkl ✗")
        
        if os.path.exists('vectorizer.pkl'):
            st.success("vectorizer.pkl ✓")
        else:
            st.error("vectorizer.pkl ✗")
    
    st.markdown("---")
    st.markdown("**Pour utiliser votre modèle :**")
    st.markdown("""
    1. Sauvegardez votre modèle :
    ```python
    import joblib
    joblib.dump(model, 'modele_sentiment.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    ```
    
    2. Placez les fichiers dans le même dossier que app.py
    3. Redémarrez l'application
    """)