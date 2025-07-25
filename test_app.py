from flask import Flask, render_template

app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

class DummyUser:
    is_authenticated = False
    username = "demo_user"

@app.route('/')
def index():
    return render_template('main.html', current_user=DummyUser())

@app.route('/predict', methods=['POST'])
def predict():
    return "Prediction functionality will be available soon!"

@app.route('/dashboard')
def dashboard():
    return "Dashboard functionality will be available soon!"

@app.route('/financial_literacy')
def financial_literacy():
    return "Financial literacy section will be available soon!"

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/logout')
def logout():
    return "Logout functionality will be available soon!"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
