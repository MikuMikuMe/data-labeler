# data-labeler

Creating an automated tool for precise labeling and classification of unstructured data is an interesting project. Below is a simplified version of a Python program for a data-labeling tool. We'll use a basic text classification example using the natural language processing library, `nltk`, and a simple machine learning model from `sklearn`.

```python
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess text data
def preprocess_text(text):
    try:
        tokens = word_tokenize(text)
        tokens = [word.lower() for word in tokens if word.isalpha()]  # Remove punctuation and lowercase
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
        return ' '.join(filtered_tokens)
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return ''

# Sample Data - In real-world scenarios, this would be replaced with your dataset
data = {
    'text': [
        "Machine learning is fascinating.",
        "Python programming is amazing!",
        "Data science includes statistics.",
        "I love watching movies.",
        "Mathematics is essential for ML."
    ],
    'label': ['technology', 'technology', 'technology', 'entertainment', 'technology']
}

# Convert data to pandas DataFrame
df = pd.DataFrame(data)

# Preprocess text data
df['text'] = df['text'].apply(preprocess_text)

try:
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

    # Create a pipeline with text preprocessing and classifier
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())

    # Train the model
    model.fit(X_train, y_train)

    # Predict the labels for test data
    predicted = model.predict(X_test)

    # Print classification report
    print("Classification report:")
    print(classification_report(y_test, predicted))

    # Print accuracy
    accuracy = accuracy_score(y_test, predicted)
    print(f"Accuracy: {accuracy:.2f}")

except Exception as e:
    print(f"An error occurred during processing: {e}")

```

### Explanation:

1. **Data Preprocessing**: We've added a `preprocess_text` function to clean and preprocess the input data. This is essential for handling unstructured text data.

2. **Data**: We've used a small set of sample data. In practice, you would use a larger dataset that accurately reflects your problem space.

3. **Error Handling**: Basic try-except blocks are included to catch errors in key steps, which may occur due to incorrect data formats or processing issues.

4. **Model**: A `MultinomialNB` model is used for classification. Though this model is suitable for simple text classification tasks, you might consider more advanced models like `RandomForest` or neural networks for more complex datasets.

5. **Pipeline**: The `make_pipeline` function simplifies the application of preprocessing and modeling steps. Adjust this as needed depending on your text specific requirements.

6. **Output**: The program prints out the classification report and accuracy on the test data.

Remember, this code is meant to illustrate a simple and generic approach for text classification. It can be more sophisticated based on the structure of unstructured data you wish to work with, by adding more complex preprocessing, feature extraction, and model evaluation techniques.