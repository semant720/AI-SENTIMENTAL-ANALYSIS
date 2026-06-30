from flask import Flask, render_template, request
import torch
import torch.nn as nn
import joblib


# ==========================
# RNN Model
# ==========================
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.rnn(x, h0)

        out = self.fc(out[:, -1, :])

        return out


# ==========================
# Flask App
# ==========================
app = Flask(__name__)


# ==========================
# Load TF-IDF Vectorizer
# ==========================
vectorizer = joblib.load("tfidf_vectorizer.pkl")


# ==========================
# Load Trained Model
# ==========================
model = torch.load(
    "sentimental_model.pth",
    map_location=torch.device("cpu"),
    weights_only=False
)

model.eval()


# ==========================
# Home Page
# ==========================
@app.route("/")
def home():
    return render_template("index.html")


# ==========================
# Prediction Route
# ==========================
@app.route("/predict", methods=["POST"])
def predict():

    review = request.form["review"]

    # TF-IDF Transformation
    vector = vectorizer.transform([review]).toarray()

    # Convert to Tensor
    vector = torch.from_numpy(vector).float()

    # Same preprocessing as training
    vector = vector.unsqueeze(1)

    with torch.no_grad():

        output = model(vector)

        probability = torch.sigmoid(output.squeeze()).item()

    if probability >= 0.5:
        prediction = "😊 Positive"
    else:
        prediction = "😞 Negative"

    return render_template(
        "index.html",
        prediction=prediction,
        probability=f"{probability*100:.2f}%",
        review=review
    )


# ==========================
# Run App
# ==========================
if __name__ == "__main__":
    app.run(debug=True)