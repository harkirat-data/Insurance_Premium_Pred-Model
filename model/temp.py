import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

preprocessor = model.named_steps["preprocessor"]

encoder = preprocessor.named_transformers_["cat"]

print("Categorical columns:")
print(["age_group", "lifestyle_risk", "occupation", "city_tier"])

print("\nCategories:")
for col, cats in zip(
    ["age_group", "lifestyle_risk", "occupation", "city_tier"],
    encoder.categories_
):
    print(f"{col}: {list(cats)}")
    