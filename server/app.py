from flask import Flask

app=Flask(__name__)

@app.route('/')
def home():
    return 'server is alive'

app.run(debug=True)