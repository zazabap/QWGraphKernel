
from ucimlrepo import fetch_ucirepo

# fetch dataset
pen_based_recognition_of_handwritten_digits = fetch_ucirepo(id=81)

# data (as pandas dataframes)
X = pen_based_recognition_of_handwritten_digits.data.features
y = pen_based_recognition_of_handwritten_digits.data.targets

print(X[:10])
print(y[:10])

# metadata
print(pen_based_recognition_of_handwritten_digits.metadata)

# variable information
print(pen_based_recognition_of_handwritten_digits.variables)

