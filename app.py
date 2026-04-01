# =====================================
# 1. IMPORTS
# =====================================
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# =====================================
# 2. LOAD MODEL
# =====================================
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# =====================================
# 3. DATASET (UPDATED WITH BETTER COVERAGE)
# =====================================
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

# Personal Care
("Nail Treatments","Personal Care"),
("Hair Shampoo","Personal Care"),
("Hair Conditioner","Personal Care"),
("Hair Styling","Personal Care"),
("Hair Straightening Cream","Personal Care"),
("Hair Treatments","Personal Care"),
("Adult Incontinence","Personal Care"),
("Face Care","Personal Care"),

# Home Care
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

# Other
("Kits","Other"),

# Personal Care (extra coverage)
("Toothpaste","Personal Care"),
("Soap Bar","Personal Care"),
("Body Lotion","Personal Care"),
("Hand Sanitizer","Personal Care"),
("Face Cream","Personal Care"),

# Home Care (extra coverage)
("Dishwashing Liquid","Home Care"),
("Glass Cleaner","Home Care"),
("Garbage Bags","Home Care"),
("Air Freshener","Home Care"),
("Insect Spray","Home Care"),

# Drinks (extra coverage)
("Mineral Water","Drinks"),
("Soft Drink","Drinks"),
("Juice","Drinks"),
("Milkshake","Drinks"),

# 🔥 NEW FIX FOR SMOOTHIES / SHAKES
("Fruit Smoothie","Drinks"),
("Banana Smoothie","Drinks"),
("Strawberry Shake","Drinks"),
("Protein Shake","Drinks"),

# Food (extra coverage)
("Potato Chips","Food"),
("Bread","Food"),
("Rice","Food"),
("Instant Noodles","Food"),
]

df = pd.DataFrame(data, columns=["text","label"])

# =====================================
# 4. LABEL ENCODING
# =====================================
le = LabelEncoder()
y = le.fit_transform(df["label"])

# =====================================
# 5. ADD CONTEXT
# =====================================
texts = [f"{t} is a product category" for t in df["text"]]

# =====================================
# 6. EMBEDDINGS
# =====================================
X = model.encode(texts)

# =====================================
# 7. TRAIN MODEL
# =====================================
clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X, y)

# =====================================
# 8. PREDICT FUNCTION
# =====================================
def predict(text):
    emb = model.encode([f"{text} is a product category"])
    pred = clf.predict(emb)[0]
    return le.inverse_transform([pred])[0]

# =====================================
# 9. UI
# =====================================
st.title("🧠 Product Category Classifier")

user_input = st.text_input("Enter product name:")

if st.button("Predict"):
    if user_input:
        result = predict(user_input)
        st.success(f"{user_input} → {result}")
    else:
        st.warning("Enter something")

# =====================================
# 10. QUICK TEST
# =====================================
st.markdown("### 🔍 Quick Test")

tests = [
    "Banana Smoothie",
    "Mango Smoothie",
    "Protein Shake",
    "Mineral Water",
    "Toothpaste",
    "Laundry Bleach"
]

for t in tests:
    if st.button(t):
        st.success(f"{t} → {predict(t)}")