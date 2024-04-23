from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mysql.connector as mq
from mysql.connector import Error
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np
import matplotlib
matplotlib.use('agg')  # Set the backend to 'agg' before importing pyplot
import matplotlib.pyplot as plt

app = Flask(__name__)

def dbconnection():
    con = mq.connect(host='localhost', database='diabetis',user='root',password='root')
    return con

# Load dataset
df = pd.read_csv('diabetes_pres.csv')

# Preprocess prescription data (e.g., lowercase, remove stopwords, etc.)
# Implement your preprocessing steps here

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
prescription_tfidf = tfidf_vectorizer.fit_transform(df['Prescription/Treatment'].astype(str))

# Calculate cosine similarity matrix
cosine_sim = cosine_similarity(prescription_tfidf, prescription_tfidf)

def recommend_similar_prescriptions(age, gender, weight, height, bmi, fbs,hba1c, total_cholesterol, hdl_cholesterol, ldl_cholesterol, triglycerides, systolic_bp, diastolic_bp, symptoms):
    # Combine input data into a single string (you can customize this)
    input_data = ' '.join([str(age), gender, str(weight), str(height), str(bmi), str(fbs),str(hba1c), str(total_cholesterol), str(hdl_cholesterol), str(ldl_cholesterol), str(triglycerides), str(systolic_bp), str(diastolic_bp), symptoms])
    
    # TF-IDF Vectorization for input data
    input_tfidf = tfidf_vectorizer.transform([input_data])

    # Calculate cosine similarity with input data
    sim_scores = list(enumerate(cosine_similarity(input_tfidf, prescription_tfidf)[0]))

    # Sort prescriptions based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top similar prescriptions
    top_similar_prescriptions_indices = [x[0] for x in sim_scores[:5]]  # Change 5 to the number of recommendations you want
    
    # Get recommended prescriptions
    recommended_prescriptions = df.iloc[top_similar_prescriptions_indices][['Prescription/Treatment', 'Dosage', 'Frequency', 'Duration (weeks)']].values.tolist()
    
    return recommended_prescriptions

@app.route('/')
def index():
    return render_template('index.html')


def cluster():
    # Read CSV data
    data = pd.read_csv("diabetes_pres.csv")
    
    # Preprocessing (if needed)
    # Example: scaling numerical features
    numerical_features = ['Age', 'Weight (kg)', 'Height (cm)', 'BMI', 'FBS (mg/dL)', 'HbA1c (%)', 'Total Cholesterol (mg/dL)', 'HDL Cholesterol (mg/dL)', 'LDL Cholesterol (mg/dL)', 'Triglycerides (mg/dL)', 'Systolic BP (mmHg)', 'Diastolic BP (mmHg)']
    data[numerical_features] = data[numerical_features].apply(lambda x: (x - x.mean()) / x.std())

    # Clustering
    kmeans = KMeans(n_clusters=3)
    data['cluster'] = kmeans.fit_predict(data[numerical_features])

    # Visualizing clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(data['Age'], data['Weight (kg)'], c=data['cluster'], cmap='viridis')
    plt.xlabel('Age')
    plt.ylabel('Weight (kg)')
    plt.title('Cluster Visualization')
    plt.grid(True)
    
    # Save the plot as an image
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    # Convert the image to base64
    encoded = base64.b64encode(image_png).decode("utf-8")
    image = "data:image/png;base64," + encoded
    return image

@app.route('/recommendation', methods=['POST'])
def recommendation():
    age = int(request.form['age'])
    gender = request.form['gender']
    weight = int(request.form['weight'])
    height = int(request.form['height'])
    bmi = float(request.form['bmi'])
    fbs = int(request.form['fbs'])
    hba1c = float(request.form['hba1c'])
    total_cholesterol = int(request.form['total_cholesterol'])
    hdl_cholesterol = int(request.form['hdl_cholesterol'])
    ldl_cholesterol = int(request.form['ldl_cholesterol'])
    triglycerides = int(request.form['triglycerides'])
    systolic_bp = int(request.form['systolic_bp'])
    diastolic_bp = int(request.form['diastolic_bp'])
    symptoms = request.form['symptoms']

    recommendations = recommend_similar_prescriptions(age, gender, weight, height, bmi, fbs,hba1c, total_cholesterol, hdl_cholesterol, ldl_cholesterol, triglycerides, systolic_bp, diastolic_bp, symptoms)
    con = dbconnection()
    cursor = con.cursor()
    cursor.execute("insert into patients(age,gender,weight,height,bmi,fbs,hba1c,total_cholesterol,hdl_cholesterol,ldl_cholesterol,triglycerides,systolic_bp,diastolic_bp,symptoms) values('{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}')".format(
        age,gender,weight,height,bmi,fbs,hba1c,total_cholesterol,hdl_cholesterol,ldl_cholesterol,triglycerides,systolic_bp,diastolic_bp,symptoms))
    con.commit()

    # Close connection
    con.close()
    image = cluster()
    return render_template('recommendation.html', recommendations=recommendations,cluster_image=image)

if __name__ == '__main__':
    app.run(debug=True)
