from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__)

@app.route('/')
@app.route('/login')
def homes():
    msg = ''
    return render_template('login.html', msg = msg)

# @app.route('/home')
# def home():
#     msg = ''
#     return render_template('home.html', msg = msg)

if __name__== "__main__" :
    app.run(debug = True)
