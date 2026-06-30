# AI Sentiment Analysis

An AI-powered Sentiment Analysis web application built using **Flask** and **PyTorch**. The application analyzes user-entered text and predicts whether the sentiment is **Positive**, **Negative**, or **Neutral** using a trained deep learning model.

---

## 🚀 Features

- Analyze text sentiment instantly.
- Predicts:
  - 😊 Positive
  - 😞 Negative
  - 😐 Neutral
- Clean and responsive web interface.
- Uses a trained PyTorch model.
- TF-IDF vectorization for text preprocessing.
- Simple Flask backend.

---

## 🛠️ Tech Stack

### Frontend
- HTML5
- CSS3
- JavaScript

### Backend
- Flask
- Python

### Machine Learning
- PyTorch
- Scikit-learn
- TF-IDF Vectorizer
- NumPy
- Pandas

---

## 📁 Project Structure

```
sentiment720/
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── templates/
│   └── index.html
│
├── app.py
├── sentimental_model.pth
├── tfidf_vectorizer.pkl
├── requirements.txt
├── Procfile
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/sentiment720.git
cd sentiment720
```

### 2. Create a Virtual Environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python app.py
```

The application will start on:

```
http://127.0.0.1:5000
```

---

## 🧠 How It Works

1. User enters text into the web page.
2. The input text is cleaned and preprocessed.
3. TF-IDF converts the text into numerical features.
4. The trained PyTorch model predicts the sentiment.
5. The predicted result is displayed to the user.

---

## 📂 Files Description

| File | Description |
|------|-------------|
| `app.py` | Flask application |
| `sentimental_model.pth` | Trained PyTorch sentiment model |
| `tfidf_vectorizer.pkl` | Saved TF-IDF vectorizer |
| `templates/` | HTML templates |
| `static/` | CSS, JavaScript, and images |
| `requirements.txt` | Python dependencies |

---

## 📸 Screenshots

Add screenshots of your application here.

Example:

```
screenshots/
    home.png
    result.png
```

---

## 🔮 Future Improvements

- Emotion Detection
- Multi-language Support
- Real-time Sentiment Analysis
- Voice Input Support
- Better UI/UX
- Model Performance Improvements

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository.
2. Create a new branch.

```bash
git checkout -b feature-name
```

3. Commit your changes.

```bash
git commit -m "Add new feature"
```

4. Push the branch.

```bash
git push origin feature-name
```

5. Open a Pull Request.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Semant Kansal**

GitHub: https://github.com/your-github-username
