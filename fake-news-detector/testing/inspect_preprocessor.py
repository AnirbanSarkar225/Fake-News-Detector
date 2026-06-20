import joblib

preprocessor = joblib.load("model/preprocessor.pkl")

print(type(preprocessor))
print("\nAvailable methods:\n")
print(dir(preprocessor))