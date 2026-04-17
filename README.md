# 🛒 Product Category Classification using BERT + LLM

## 📌 Overview
This project builds an AI pipeline to classify products into categories such as Food, Drinks, Home Care, Personal Care, etc.

It combines:
- BERT embeddings (semantic similarity)
- LLM (FLAN-T5) classification

---

## ⚙️ Pipeline

1. Input: Product name + description  
2. BERT computes similarity with category definitions  
3. LLM predicts category using prompt  
4. Compare both outputs  
5. If both agree → final prediction  
6. If not → marked as conflict  

---

## 📊 Results

| Model | Accuracy |
|------|--------|
| BERT | 88.54% |
| FLAN (LLM) | 84.71% |

👉 **Consensus Accuracy (when both agree): 94.57%**

👉 **82% of products were auto-classified without conflict**

---

## 🔍 Key Insights

- LLM understands semantics better than embeddings in many cases  
- BERT performs better on structured categories (Home Care, Personal Care)  
- LLM tends to predict "Other" when uncertain  
- Combining both improves reliability significantly  

---

## ⚠️ Limitations

- LLM does not provide real confidence scores  
- BERT suffers from category overlap (Food vs Drinks)  
- "Other" category is difficult for both models  
- Some ground truth labels in dataset are incorrect  

---

## 📈 Error Analysis

Conflicts between models fall into 3 types:

1. BERT correct, LLM predicts "Other"  
2. LLM correct, BERT confused due to overlapping definitions  
3. Truly ambiguous products  

---

## 🚀 Future Improvements

- Replace FLAN with stronger LLM (Mistral / GPT)  
- Use margin-based confidence for BERT  
- Improve category definitions  
- Train custom classifier (SetFit)  

---

## 🛠 Tech Stack

- Python  
- Sentence Transformers  
- HuggingFace Transformers  
- Scikit-learn  

---

## 👤 Author

Muhammad Ashir