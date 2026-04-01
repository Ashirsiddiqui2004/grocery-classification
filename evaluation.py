import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# =========================
# DATA
# =========================
data = [
("NB Fresh Turkey","Food"),
("NB Fresh Fruits","Food"),
("Dairy Beverage with Juice","Drinks"),
("UHT Milk","Food"),
("Powder Milk","Food"),
("Drinkable Yoghurt","Food"),
("Ice Creams","Food"),
("Butter","Food"),
("Evaporated Milk","Food"),
("Sour Cream","Food"),
("Distilled","Drinks"),
("Ground Coffee","Drinks"),
("Bean Coffee","Drinks"),
("Mate Cocido","Drinks"),
("Ready To Drink Tea","Drinks"),
("RTD Sport Drinks","Drinks"),
("Ready To Drink Coffee","Drinks"),
("RTD Hot Chocolate","Drinks"),
("Other Veggie Juices","Drinks"),
("Crisps & Extruded Snacks","Food"),
("Savoury Biscuits","Food"),
("Savoury Toasts","Food"),
("Sweet Biscuits","Food"),
("Salad Dressing & Condiment","Food"),
("Sugar","Food"),
("Instant Macaroon","Food"),
("Granola","Food"),
("Other Hot Cereals","Food"),
("Olive Oil","Food"),
("Meals & Dishes","Food"),
("Canned Tuna","Food"),
("Pate","Food"),
("Dried Meat","Food"),
("Cold Meat","Food"),
("Canned Chicken","Food"),
("Industrial Bread","Food"),
("Chocolate","Food"),
("Easter Eggs","Food"),
("Hamburguers","Food"),
("Frozen Vegetables","Food"),
("Empanada Dough","Food"),

("Nail Treatments","Personal Care"),
("Hair Shampoo","Personal Care"),
("Hair Conditioner","Personal Care"),
("Hair Styling","Personal Care"),
("Hair Straightening Cream","Personal Care"),
("Hair Treatments","Personal Care"),
("Adult Incontinence","Personal Care"),
("Face Care","Personal Care"),

("Laundry Bleach","Home Care"),
("Cleaners & Disinfectants","Home Care"),
("Drain Decloggers","Home Care"),
("Cleaning Gloves","Home Care"),
("Stainless Steel Scrubbers","Home Care"),
("Laminate Floor Cleaners","Home Care"),
("Laundry Bar","Home Care"),
("Laundry Additives","Home Care"),
("Plastic Storage Bags","Home Care"),
("Insect Repellents","Home Care"),
("Scented Candles","Home Care"),
("Batteries","Home Care"),

("Kits","Other")
]

df = pd.DataFrame(data, columns=["text","label"])

# =========================
# SPLIT
# =========================
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# =========================
# ENCODING
# =========================
le = LabelEncoder()
y_train = le.fit_transform(train_df["label"])
y_test = le.transform(test_df["label"])

# =========================
# MODEL
# =========================
model = SentenceTransformer("all-MiniLM-L6-v2")

train_texts = [f"{t} is a product category" for t in train_df["text"]]
test_texts = [f"{t} is a product category" for t in test_df["text"]]

X_train = model.encode(train_texts)
X_test = model.encode(test_texts)

# =========================
# TRAIN
# =========================
clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train, y_train)

# =========================
# PREDICT
# =========================
y_pred = clf.predict(X_test)

# =========================
# ACCURACY
# =========================
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

# =========================
# CLASSIFICATION REPORT (FIXED)
# =========================
print("\nClassification Report:\n")
print(classification_report(
    y_test,
    y_pred,
    labels=range(len(le.classes_)),
    target_names=le.classes_,
    zero_division=0
))

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# =========================
# WRONG PREDICTIONS
# =========================
results = pd.DataFrame({
    "Text": test_df["text"],
    "Actual": le.inverse_transform(y_test),
    "Predicted": le.inverse_transform(y_pred)
})

print("\nPredictions:\n")
print(results)

print("\nWrong Predictions:\n")
print(results[results["Actual"] != results["Predicted"]])