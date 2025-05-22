from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from datetime import datetime
import re
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///feedback.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Load your model
model = load_model(r'C:\Users\Sravani\Downloads\rice_disease_project\rice_disease_model.h5')

# Class labels
class_names = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro']
# Disease data
disease_info = {
    'Bacterialblight': {
        'description': '1.Caused by Xanthomonas oryzae bacteria '
                       '2.Symptoms include yellowing '
                       '3.Drying of leaves.',
        'remedy': '1.Use resistant varieties, '
                  '2.Avoid over-fertilization, '
                  '3.Apply copper-based bactericides.'
    },
    'Blast': {
        'description': '1.Caused by the fungus Magnaporthe oryzae '
                       '2.It appears as spindle-shaped lesions on leaves.',
        'remedy': '1.Use resistant varieties, '
                  '2.Ensure good field drainage, '
                  '3.Apply fungicides like tricyclazole.'
    },
    'Brownspot': {
        'description': '1.Caused by Bipolaris oryzae '
                       '2.Presents as brown lesions with yellow halos.',
        'remedy': '1.Apply balanced fertilizers, '
                  '2.Use disease-free seeds, '
                  '3.Spray fungicides such as Mancozeb.'
    },
    'Tungro': {
        'description': '1.A viral disease transmitted by leafhoppers; '
                       '2.Plants show stunting and yellow-orange discoloration.',
        'remedy': '1.Control vector population with insecticides, '
                  '2.Use tungro-resistant varieties.'
    }
}

# Feedback model
class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(120))
    phone = db.Column(db.String(15)) 
    feedback = db.Column(db.Text)
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)

# Home page: Upload + Predict
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None
    recommendation = None
    confidence = None
    timestamp = None

    if request.method == 'POST':
        img_file = request.files['file']
        if img_file:
            filename = secure_filename(img_file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img_file.save(file_path)
            image_path = file_path

            img = image.load_img(file_path, target_size=(128, 128))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            pred = model.predict(img_array)
            predicted_class = class_names[np.argmax(pred)]
            prediction = predicted_class
            confidence = float(np.max(pred)) * 100
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return render_template('index.html',
                           prediction=prediction,
                           confidence=confidence,
                           image_path=image_path,
                           recommendation=recommendation,
                           timestamp=timestamp)

# Disease details page
@app.route('/details', methods=['POST'])
def details():
    prediction = request.form.get('prediction')
    confidence = request.form.get('confidence')

    info = disease_info.get(prediction)
    if info:
        description_lines = re.split(r'(?=\d\.)', info['description'])
        remedy_lines = re.split(r'(?=\d\.)', info['remedy'])
        recommendation = {
            'description': [line.strip() for line in description_lines if line.strip()],
            'remedy': [line.strip() for line in remedy_lines if line.strip()]
        }
    else:
        recommendation = {'description': ['No information available.'], 'remedy': ['No remedy available.']}

    return render_template('details.html',
                           prediction=prediction,
                           confidence=confidence,
                           recommendation=recommendation)

# Feedback form page
@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        feedback_text = request.form['feedback']

        new_feedback = Feedback(name=name, email=email, phone=phone,feedback=feedback_text)
        db.session.add(new_feedback)
        db.session.commit()

        # Show thank you page after submission
        return render_template('feedback_success.html', name=name)

    return render_template('feedback.html')


@app.route('/view-feedback')
def view_feedback():
    feedbacks = Feedback.query.all()
    return render_template('view_feedback.html', feedbacks=feedbacks)
@app.route('/clear-feedback')
def clear_feedback():
    Feedback.query.delete()
    db.session.commit()
    return "All feedback has been deleted."



# Run the app
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)

