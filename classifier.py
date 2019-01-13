import os
import io
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def read_files(path):
    for dir_path, dir_names, file_names in os.walk(path):
        for file_name in file_names:
            path = os.path.join(dir_path, file_name)
            in_body = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if in_body:
                    lines.append(line)
                elif line == '\n':
                    in_body = True
            f.close()
            message = '\n'.join(lines)
            yield path, message

def data_frame_from_directory(path, classification):
    rows = []
    index = []
    for file_name, message in read_files(path):
        rows.append({'message': message, 'class': classification})
        index.append(file_name)

    return DataFrame(rows, index=index)

data = DataFrame({'message': [], 'class': []})

data = data.append(data_frame_from_directory('./emails/spam', 'spam'), sort=True)
data = data.append(data_frame_from_directory('./emails/ham', 'ham'), sort=True)

print("--- Data Summary ---")
print(data.head())

# Split each message into a list of words and fit them in a MultinomialNB classifier
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['message'].values)
classifier = MultinomialNB()
targets = data['class'].values
classifier.fit(counts, targets)

# Predict if spam or not using the classifier
examples = ['Free Viagra now!!!', "Hi Bob, how about a game of golf tomorrow?"]
example_counts = vectorizer.transform(examples)
predictions = classifier.predict(example_counts)

# Print results
print("--- Results ---")
for index, prediction in enumerate(predictions):
    print(index, prediction)