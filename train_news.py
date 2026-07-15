# train_news.py
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Nepali Multi-class Dataset (News Headlines & Snippets)
news_data = {
    "text": [
        # Politics (राजनीति)
        "प्रधानमन्त्रीले संसदमा सम्बोधन गर्दै नयाँ बजेटको घोषणा गर्नुभयो।",
        "निर्वाचन आयोगले आगामी चुनावको तयारी तीव्र पारेको छ।",
        "विपक्षी दलहरूले सरकारको नयाँ नीतिको कडा विरोध गरेका छन्।",
        "मन्त्रिपरिषद्को बैठकले नयाँ कानुन पारित गर्ने निर्णय गर्यो।",
        
        # Sports (खेलकुद)
        "नेपाली राष्ट्रिय क्रिकेट टोलीले ऐतिहासिक खेलमा विजय हासिल गर्यो।",
        "काठमाडौँमा आयोजना हुने फुटबल प्रतियोगिताको खेल तालिका सार्वजनिक।",
        "नेपालका लेग स्पिनरले उत्कृष्ट बलिङ गर्दै विपक्षी टोलीलाई धराशयी बनाए।",
        "ओलम्पिक खेलकुदमा सहभागिता जनाउन नेपाली खेलाडीहरू प्रस्थान गरे।",
        
        # Entertainment (मनोरञ्जन)
        "नयाँ नेपाली चलचित्रको ट्रेलरले युट्युबमा निकै राम्रो दर्शक पाएको छ।",
        "लोकप्रिय गायकको नयाँ गीत सार्वजनिक समारोहमा धेरै कलाकारहरूको उपस्थिति।",
        "चलचित्र पत्रकार संघले उत्कृष्ट अभिनेता र अभिनेत्रीलाई अवार्ड प्रदान गर्यो।",
        "यो हप्ता रिलिज भएको नयाँ सिनेमाले बक्स अफिसमा राम्रो ओपनिङ गर्यो।",
        
        # Technology (प्रविधि)
        "नेपाल दूरसञ्चार प्राधिकरणले देशैभरि फाइभ-जी सेवा विस्तार गर्ने योजना बनायो।",
        "नयाँ स्मार्टफोन बजारमा सार्वजनिक भयो, यसमा उत्कृष्ट क्यामेरा रहेको छ।",
        "आर्टिफिसियल इन्टेलिजेन्स (AI) ले नेपाली सफ्टवेयर उद्योगमा क्रान्ति ल्याउँदैछ।",
        "साइबर सुरक्षा बढाउन सरकारले नयाँ प्रविधि र सफ्टवेयर लागू गर्ने भएको छ।"
    ],
    # 0: Politics, 1: Sports, 2: Entertainment, 3: Technology
    "category": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
}

df_news = pd.DataFrame(news_data)

# 2. Split into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(
    df_news["text"], 
    df_news["category"], 
    test_size=0.25, 
    random_state=42, 
    stratify=df_news["category"] # Ensures equal category distribution in train/test
)
# 3. Create the Machine Learning Pipeline (Warnings Fixed!)
news_pipeline = make_pipeline(
    TfidfVectorizer(
        ngram_range=(1, 2), 
        token_pattern=r"(?u)\b\w+\b"
    ),
    # Removed multi_class="multinomial" to fix the FutureWarning
    LogisticRegression(solver="lbfgs") 
)

# 4. Train the Model
print("Training the Nepali News Multi-class Classifier...")
news_pipeline.fit(X_train, y_train)
print("Training Completed Successfully!")

# 5. Quick Evaluation
predictions = news_pipeline.predict(X_test)
print("\n📊 Model Evaluation Report:")
# Added zero_division=0 to handle small test set math warnings cleanly
print(classification_report(
    y_test, 
    predictions, 
    target_names=["Politics", "Sports", "Entertainment", "Tech"],
    zero_division=0
))


# 6. Save the Model and Metadata to a Pickle File
news_model_artifact = {
    "pipeline": news_pipeline,
    "categories": {0: "राजनीति (Politics)", 1: "खेलकुद (Sports)", 2: "मनोरञ्जन (Entertainment)", 3: "प्रविधि (Tech)"}
}

with open("nepali_news_model.pkl", "wb") as f:
    pickle.dump(news_model_artifact, f)

print("\n🎉 Model successfully saved as 'nepali_news_model.pkl'!")
