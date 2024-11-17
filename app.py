from flask import Flask, render_template, request
import pickle

# Open the tokenizer and model:
tokenizer = pickle.load(open("models/cv.pkl", "rb"))
model = pickle.load(open("models/clf.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")
  
# Making a POST request to predict route
@app.route("/predict", methods = ["POST"])
def predict():
    # Get email-content from form and store it in the email_text variable
    email = request.form.get("content")
        
    # Use the tokenizer on the email_text and store in the tokenized_email variable
    tokenized_email = tokenizer.transform([email])
    
    # Use the model to predict based on the tokenized_email
    prediction = model.predict(tokenized_email)
    
    # If the tokenized email is spam, predict as 1 else predict as -1:
    prediction = 1 if prediction == 1 else -1
    
    
    
    return render_template("index.html", prediction = prediction, email_text = email) 
  


if __name__ == "__main__":
    app.run(host = "0.0.0.0", debug = True, port = 8080)
    
# Functions to create:

# Function that connects to s3 bucket:
def s3():
    return 0
    
    

































