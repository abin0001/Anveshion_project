from flask import Flask,render_template,request, redirect, url_for, session,jsonify
import pymysql
import re
import bcrypt
import json
from IPython.display import display
from IPython.display import Markdown
import textwrap
from keras.models import load_model
from keras.preprocessing import image
import google.generativeai as genai
import cv2 as cv
import tensorflow as tf
import numpy as np 
import os
from keras.applications.vgg16 import preprocess_input


dic = {0 : 'one', 1 : 'two',2 : 'three', 3 : 'four', 4 : 'five', 5 : 'six', 6 : 'seven', 7 : 'eight',8 : 'nine'}


model = load_model('my_model.keras')
model.make_predict_function()

def predict_label(img_path):
	img=cv.imread(img_path)
	r_img=np.array([cv.resize(img,(224,224))])
	p = model.predict(r_img)
	classes = np.argmax(p,axis=1)
	return dic[classes[0]]


API_KEY='AIzaSyC73w6AC-OpXFUyBHbBmLjTLHH6TBimENE'
genai.configure(api_key=API_KEY)
def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


def respi_gen(promt):
    model = genai.GenerativeModel('gemini-pro') 
    response = model.generate_content(promt, stream=True)
    response.resolve()
    response = response.text
    return response

app = Flask(__name__)
app.secret_key = 'hackerhello'  

mydb = pymysql.connect(
    host='127.0.0.1',
    user="root",
    password="",
    database="userdata"
)

mycursor = mydb.cursor()
@app.route('/')
def Home():
	return render_template('land.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password'].encode('utf-8')

        sql = "SELECT password FROM users WHERE username = %s"
        mycursor.execute(sql, (username,))
        result = mycursor.fetchone()

        if result:
            hashed_password = result[0].encode('utf-8')
            if bcrypt.checkpw(password, hashed_password):
                session['logged_in'] = True
                session['username'] = username
                return redirect(url_for('dashboard'))
            else:
                return 'Invalid password'
        else:
            return 'Invalid username'

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password'].encode('utf-8')
        sql = "SELECT * FROM users WHERE username = %s"
        mycursor.execute(sql, (username,))
        result = mycursor.fetchone()

        if result:
            return 'Username already exists'
        hashed_password = bcrypt.hashpw(password, bcrypt.gensalt())

        sql="INSERT INTO users (username, password) VALUES (%s,%s)"
        mycursor.execute(sql,(username,hashed_password))
        mydb.commit()

        return redirect('login')

    return render_template('register.html')



@app.route('/dashboard')
def dashboard():
    if 'logged_in' in session:
        return render_template('bpm.html')
    else:
        return redirect('/login') 
    
@app.route('/recomend')
def recomend():
    if 'logged_in' in session:
        return render_template('recomend.html')
    else:
        return redirect('/login') 

@app.route('/chat')
def chat():
    if 'logged_in' in session:
        return render_template('bot.html')
    else:
        return redirect('/login') 
    
@app.route("/get", methods=["GET", "POST"])
def chatbot():
    if request.method == "POST":
        prompt = request.form["msg"]
        response =respi_gen(prompt)
        return response
    
@app.route('/herblens')
def herb():
    if 'logged_in' in session:
        return render_template('upload.html')
    else:
        return redirect('/login')

@app.route("/upload", methods=['GET', 'POST'])
def kuch_bhi():
	return render_template("upload.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_hours():
	if request.method == 'POST':
		img = request.files['my_image']
		img_path = os.path.join('static',img.filename)
		img.save(img_path) 
		p = predict_label(img_path)
	return render_template("upload.html", prediction = p, img_path = img_path)

    
if __name__ == '__main__':

	app.run(debug=True)
