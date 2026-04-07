# Import required libraries
import nltk  # used for text processing
import string  # used to remove punctuation marks

# Import stopwords (common words like "is", "the")
from nltk.corpus import stopwords

# Import TF-IDF and cosine similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords (only first time)
nltk.download('stopwords')

# Function to clean text
def preprocess(text):
    text = text.lower()  # convert all text to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  
    # removes punctuation like .,!? 

    words = text.split()  # split sentence into words

    # remove common stopwords
    words = [word for word in words if word not in stopwords.words('english')]

    return " ".join(words)  # join words back into sentence


# Take input from user
resume = input("Enter your resume text:\n")
job_desc = input("\nEnter job description:\n")

# Clean both texts
resume_clean = preprocess(resume)
job_desc_clean = preprocess(job_desc)

# Convert text into numbers using TF-IDF
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([resume_clean, job_desc_clean])

# Calculate similarity
similarity = cosine_similarity(vectors[0:1], vectors[1:2])

# Convert to percentage
score = similarity[0][0] * 100

# Print result
print(f"\nMatch Score: {score:.2f}%")