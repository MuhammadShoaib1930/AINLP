import os
import pickle
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# File paths
data_file = "data.txt"
vectorized_file = "vectorized_data.pkl"

# Load or process data from file
def load_context(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Please create the file and add your context.")
    
    with open(file_path, "r", encoding="utf-8") as file:
        context = file.read().strip()
    
    return context

# Save vectorized data for future use
def save_vectorized_data(data, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

# Load vectorized data
def load_vectorized_data(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

# Preprocess and tokenize data
def preprocess_context(context):
    vectorizer = TfidfVectorizer(stop_words="english")
    tokenized_data = vectorizer.fit_transform([context])
    return {"vectorizer": vectorizer, "tokenized_data": tokenized_data}

# Main function
def main():
    print("Loading Question-Answering Pipeline...")
    try:
        nlp = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    except Exception as e:
        print("Error loading the model. Ensure you have internet for the first run to cache the model.")
        raise e

    # Load context
    if os.path.exists(vectorized_file):
        print("Loading preprocessed data...")
        data = load_vectorized_data(vectorized_file)
        vectorizer, tokenized_data = data["vectorizer"], data["tokenized_data"]
    else:
        print("Processing context data...")
        context = load_context(data_file)
        data = preprocess_context(context)
        save_vectorized_data(data, vectorized_file)
        vectorizer, tokenized_data = data["vectorizer"], data["tokenized_data"]

    context = load_context(data_file)
    
    print("\nQuestion-Answering AI")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Enter your question: ").strip()

        # Exit condition
        if question.lower() == "exit":
            print("Goodbye!")
            break

        try:
            # Generate answer
            result = nlp(question=question, context=context)

            # Check for low-confidence answers
            if result['score'] < 0.3:
                print(f"\nQuestion: {question}")
                print("Answer: I'm not confident enough to answer that.\n")
            else:
                print(f"\nQuestion: {question}")
                print(f"Answer: {result['answer']}\n")
        except Exception as e:
            print("Error processing the question or context. Please try again.")
            print(f"Details: {e}\n")


if __name__ == "__main__":
    main()
