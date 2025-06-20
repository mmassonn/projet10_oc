import streamlit as st
import requests
import json

# Set the URL of the Azure function
azure_url = "https://webrecommender.azurewebsites.net/api/recommandationfunction?"

st.title("Recommandation des articles")

# Input for user ID
user_id_input = st.number_input("Entrez un numéro d'utilisateur :", min_value=1, value=1220)

# Button to trigger the recommendation
if st.button("Description"):
    try:
        # Set the parameters of the request
        request_params = {"user_id": user_id_input}

        # Send the request to the Azure function
        with st.spinner("Récupération des recommandations..."):
            r = requests.post(azure_url, params=request_params)
            r.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        # Grab the recommendations as a Python dictionary
        recommendations_dict = json.loads(r.content.decode())

        # Display the recommendations
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