import nltk
import random
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Download the required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Sample data: Define some job-related keywords for classification
qualified_keywords = [
    "experienced", "knowledgeable", "proficient", "skilled", "certified",
    "expert", "qualified", "capable", "efficient", "accomplished"
]
not_qualified_keywords = [
    "inexperienced", "unqualified", "inefficient", "novice", "beginner",
    "incompetent", "unskilled", "untrained", "inadequate", "unqualified"
]

# Function to preprocess the text data
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Tokenize the text
    words = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Generate training data
data = []
labels = []

# Add positive samples (Qualified candidates)
for keyword in qualified_keywords:
    data.append(preprocess_text(keyword))
    labels.append(1)

# Add negative samples (Not Qualified candidates)
for keyword in not_qualified_keywords:
    data.append(preprocess_text(keyword))
    labels.append(0)

# Shuffle the data
data_labels = list(zip(data, labels))
random.shuffle(data_labels)
data, labels = zip(*data_labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Function to predict candidate's qualification
def predict_candidate_qualification(text):
    preprocessed_text = preprocess_text(text)
    text_tfidf = vectorizer.transform([preprocessed_text])
    prediction = svm_classifier.predict(text_tfidf)[0]
    if prediction == 1:
        return "Qualified"
    else:
        return "Not Qualified"

# Example usage:
candidate_description = "I am an experienced and certified professional in this field."
qualification_prediction = predict_candidate_qualification(candidate_description)
print("Candidate Qualification Prediction:", qualification_prediction)
