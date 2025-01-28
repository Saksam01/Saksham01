# 📞 Real-Time AI Sales Intelligence and Dynamic Deal Recommendation System 🎯

## 🚀 Project Overview
This project is a **real-time AI-driven sales assistant** that enhances customer interactions by leveraging **sentiment analysis, intent detection, and personalized deal recommendations**. It integrates **speech recognition, NLP, and generative AI** to analyze conversations and provide actionable insights for better sales and negotiation strategies.

Built with **Streamlit**, this application provides a user-friendly **voice-based** interface for seamless interaction and real-time processing.

---

## ✨ Features

✅ **🎤 Real-Time Voice Input** – Captures live user responses using speech recognition.  
✅ **📊 Sentiment & Intent Analysis** – Determines the emotional tone and user intent from conversations.  
✅ **🔍 Tone Analysis** – Evaluates speech attributes such as confidence, hesitation, and urgency.  
✅ **📌 Personalized Recommendations** – Provides **10 actionable insights** based on user complaints.  
✅ **💼 Smart Deal Recommendations** – Suggests **5 deal strategies** to close sales more effectively.  
✅ **📝 Post-Call Summary** – Generates a **concise 3-line summary** for record-keeping.  
✅ **📂 CRM Integration** – Saves all interactions to an **Excel-based CRM system (CC.xlsx)** for easy tracking.  
✅ **🗣️ Text-to-Speech (TTS)** – Uses `pyttsx3` to **speak prompts aloud**, improving accessibility.  
✅ **🎨 Streamlit UI** – Provides an **interactive dashboard** with live voice capture and real-time results.  

---

## 🛠️ Technologies Used

### **👨‍💻 Programming Languages**
- **Python** 🐍

### **🔗 APIs & Libraries**
- **Streamlit** – Interactive UI 🎨  
- **Google Generative AI (Gemini API)** – NLP & conversation analysis 🧠  
- **SpeechRecognition** – Converts speech to text 🎤  
- **VaderSentiment** – Sentiment analysis 💬  
- **OpenPyXL** – Excel-based CRM integration 📊  
- **PyAudio** – Captures real-time audio input 🎙️  
- **pyttsx3** – Text-to-Speech (TTS) 🗣️  

---

## 📌 Installation Instructions

### ✅ **Prerequisites**
- **Python 3.8+**  
- **Pip package manager**  
- **A working microphone** 🎤  
- **Google Generative AI API key**  

### 🔧 **Installation Steps**
1️⃣ **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/ai-sales-intelligence.git
   ```
2️⃣ Navigate to the project directory

```
cd ai-sales-intelligence
```
3️⃣ Install the required dependencies
```
pip install streamlit pyaudio speechrecognition google-generativeai openpyxl vaderSentiment pyttsx3
```
4️⃣ Run the application
```
streamlit run main.py
```
##  🎯 Usage Guide
1️⃣ Click on "Start Conversation"

2️⃣ Answer the voice prompts for name, email, phone number, and company details.

3️⃣ State your complaint or request – The AI will analyze and generate recommendations.

4️⃣ View Results – Sentiment, intent, tone analysis, deal strategies, and post-call summary.

5️⃣ CRM Logging – All details are saved in CC.xlsx for future reference.

### 📊 Output Format

🖥️ Terminal / Streamlit Display

1. Sentiment Analysis
2. User Intent
3. Tone Analysis
4. Personalized Recommendations
5. Deal Recommendations
6. Post-Call Summary

##  📂 Excel File (Crm_data.xlsx)

Headers include --> Name |	Email |	Phone	Company |	Deal ID |	Date	Sentiment | Intent |	Tone |	Recommendations |	Deal Recommendations |	Post-call Summary

##  🚀 Future Enhancements


🔹 CRM API Integration – Connect with platforms like Salesforce, HubSpot 📊

🔹 Real-Time Chatbot – Text & voice-based AI assistant 💬

🔹 Multilingual Support – Handle different languages 🌍

🔹 Enhanced Deal Strategies – More advanced negotiation insights 📈

🔹 Live Dashboard Analytics – Visual representation of sales intelligence 📊

##  🤝 Contribution Guidelines

1️⃣ Fork the repository

2️⃣ Create a new branch
```
git checkout -b feature-name
```
3️⃣ Commit your changes
```
git commit -m "Add new feature"
```
4️⃣ Push and submit a Pull Request

##  🎉 Acknowledgments
🙌 Special thanks to:

1. Google Generative AI – NLP processing

2. Streamlit – User-friendly UI

3. SpeechRecognition & PyAudio – Voice capture

4. VaderSentiment – Sentiment analysis

5. OpenPyXL – Excel CRM integration

🚀 Happy Selling with AI! 🚀

















