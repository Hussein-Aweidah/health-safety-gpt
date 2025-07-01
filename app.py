import streamlit as st
from rag_pipeline import get_response

st.title("Health & Safety GPT 🇳🇿")

user_query = st.text_input("Ask your workplace safety question:")

if user_query:
    response = get_response(user_query)
    st.write("### Answer:")
    st.write(response)
