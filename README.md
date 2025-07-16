# **Ayu-Verse**

**Ayu-Verse** is an AI-powered Ayurvedic health assistant that predicts dosha types, recommends herbal remedies, and provides skin type analysis using machine learning and natural language processing. It integrates traditional Ayurvedic knowledge with modern data science to offer personalized health insights and recommendations.

---

## 🔍 **Features**

- **Dosha Prediction:** Predicts your Ayurvedic dosha (Vata, Pitta, Kapha) based on questionnaire or data input.  
- **Herbal Remedy Recommendation:** Suggests herbal remedies for various diseases and conditions using a curated dataset.  
- **Skin Type Detection:** Uses image analysis to determine skin type and provide care recommendations.  
- **Chatbot/Consultation:** Chat interface for patient-doctor interaction.  
- **User Authentication:** Supports patient and medical login/registration.  
- **Data Management:** Utilizes MongoDB for storing user and consultation data.

---

## ⚙️ **Requirements**

- **Python 3.8+**  
- See `requirements.txt` for all dependencies.

---

## 🚀 **Setup**

### 1. **Clone the repository:**
```bash
git clone https://github.com/Anand-303/Ayu-Verse.git
cd Ayu-Verse
```
### 2. **Install dependencies:**
```bash
pip install -r requirements.txt
```
### 3. **(Optional) Set up your database (MongoDB recommended).**

## 4. **Run the application:**
```bash
python app.py
```

## 📁 **File Structure**
`app.py` - Main Flask application.
`dosha_predictor.py`, `train_dosha_model.py` - Dosha prediction logic and model training.
`herbal_remedy_text_model.h5` - Pretrained model for herbal remedy recommendations.
templates/ - HTML templates for the web interface.
static/ - Static files (CSS, JS, images).
data/ - Datasets and database files.
`requirements.txt` - Python dependencies.

## 🙏 **Acknowledgements**

- Inspired by traditional Ayurvedic medicine
- Utilizes open-source datasets and machine learning libraries


---
