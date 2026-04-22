from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# Train model
data = load_iris()
X = data.data
y = data.target

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

@app.route("/")
def home():
    return "Model is running"

@app.route("/predict", methods=["POST"])
def predict():
    features = request.json["features"]
    prediction = model.predict([features])
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)