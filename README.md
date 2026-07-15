
# 📰 Nepali News Category Classifier

An interactive Natural Language Processing (NLP) application designed to classify Nepali news headlines or articles into four major categories: **Politics (राजनीति)**, **Sports (खेलकुद)**, **Entertainment (मनोरञ्जन)**, and **Technology (प्रविधि)**.

This project demonstrates **Multi-class Text Classification**, **Unicode-compatible Tokenization**, **TF-IDF Vectorization**, and interactive model evaluation with predictive probabilities using Python, Scikit-Learn, and Streamlit.

---

## 🛠️ Project Architecture & Pipeline

```text
                 User Text Input (Nepali News)
                              │
                              ▼
           TfidfVectorizer (Custom Token Pattern)
           [Preserves Nepali glyphs and halants]
                              │
                              ▼
              Multinomial Logistic Regression
           [Predicts class probability array]
                              │
                              ▼
              Interactive Confidence Breakdown
                 [Streamlit Progress Bars]

```

---

## 📂 Project Structure

```text
Nepali-News-Classifier/
│
├── nepali_news_model.pkl   # Serialized pipeline (model + custom vectorizer + category map)
├── app.py                  # Streamlit-based interactive user interface
├── train_news.py           # Multi-class machine learning training pipeline
└── README.md               # Project documentation

```

---

## 🚀 Getting Started

### 1. Installation

Clone this repository to your local machine and install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Model Training & Serialization

Run the training script to pre-process the text, train the Logistic Regression model, and generate the pickled model artifact. This script handles Unicode token boundaries and evaluates the model cleanly:

```bash
python train_news.py

```

### 3. Running the Web Application

Launch the Streamlit web dashboard to interactively test custom news inputs and inspect probability distributions:

```bash
streamlit run app.py

```

Open `http://localhost:8501` in your web browser to start classifying news!

---

## 📈 Machine Learning Implementation Details

> 💡 **Nepali NLP Note:** Standard tokenizers often break Devanagari script incorrectly. This project uses dedicated patterns to keep text features intact.

* **Unicode Token Preservation:** Built a custom token pattern `r"(?u)\b\w+\b"` to prevent Scikit-Learn's default analyzer from splitting essential Nepali vowel signs (मात्राहरू) and halants (हलन्त) into corrupt tokens.
* **Stratified Splitting:** Handled class balance during dataset splitting using `stratify=y` to ensure test-set metrics reliably evaluate all categories evenly.
* **Multinomial Logistic Regression:** Implemented multi-class classification using an optimized Softmax-like probability estimator to gauge category confidence percentage dynamically.
* **Warning-Free Compilation:** Cleanly resolved deprecation warnings regarding Scikit-Learn's updated `multi_class` parameters and mathematical division limitations on smaller test subsets.



