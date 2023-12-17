import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
with open('count_vectorizer.pkl', 'rb') as file:
    cv = pickle.load(file)
with open('spam_ham_classifier.pkl', 'rb') as file:
    model = pickle.load(file)
def main():
    st.title("Email Spam Classification")
    user_input = st.text_area("Enter your email here:")
    if st.button("Classify"):
        data = [user_input]
        vect = cv.transform(data).toarray()
        result = model.predict(vect)
        if result[0] == 0:
            st.write("This email is HAM")
        else:
            st.write("This email is SPAM")
if __name__ == "__main__":
    main()