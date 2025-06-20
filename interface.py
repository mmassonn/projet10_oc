import streamlit as st
import requests
import json

azure_url = "https://webrecommender.azurewebsites.net/api/recommandationfunction?"

st.title("Recommandation des articles")

# Afficher un menu déroulant pour la sélection de l'ID utilisateur
user_ids = list(range(1, 2001))
user_id_input = st.selectbox("Sélectionnez un numéro d'utilisateur :", user_ids)

if st.button("Description"):
    try:
        request_params = {"user_id": user_id_input}

        # Envoyer la requête
        with st.spinner("Récupération des recommandations..."):
            r = requests.post(azure_url, params=request_params)
            r.raise_for_status()

        # Mettre les recommendation dans un dictionnaire python
        recommendations_dict = json.loads(r.content.decode())

        # Afficher les recommandation
        st.subheader("Recommandations :")
        if recommendations_dict:
            st.json(recommendations_dict) # Displays the dictionary as a formatted JSON
        else:
            st.info("Aucune recommandation trouvée pour cet utilisateur.")

    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la connexion à l'API : {e}")
    except json.JSONDecodeError:
        st.error("Erreur lors du décodage de la réponse JSON. La réponse de l'API est peut-être invalide.")
    except Exception as e:
        st.error(f"Une erreur inattendue est survenue : {e}")