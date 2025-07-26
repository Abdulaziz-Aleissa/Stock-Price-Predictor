from flask import Flask, render_template, jsonify
from flask_login import LoginManager, UserMixin, current_user

app = Flask(__name__, template_folder='app/templates', static_folder='app/static')
app.config['SECRET_KEY'] = 'test-secret-key-for-demo'

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Mock user class for testing
class MockUser(UserMixin):
    def __init__(self, id):
        self.id = id
        self.username = "test_user"
        self.is_authenticated = True

class DummyUser:
    is_authenticated = False
    username = "demo_user"

@login_manager.user_loader
def load_user(user_id):
    return MockUser(user_id)

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
    # Mock paper trading data for testing
    paper_summary = type('obj', (object,), {
        'cash_balance': 100000.0,
        'total_value': 0.0,
        'total_account_value': 100000.0,
        'total_profit_loss': 0.0,
        'total_return_percent': 0.0
    })
    
    paper_portfolio = []  # Empty portfolio for testing
    
    return render_template('financial_literacy.html', 
                         paper_summary=paper_summary, 
                         paper_portfolio=paper_portfolio)

@app.route('/paper_transactions')
def paper_transactions():
    # Mock empty transactions for testing
    return jsonify([])

@app.route('/paper_buy', methods=['POST'])
def paper_buy():
    return jsonify({"status": "success", "message": "Mock buy order"})

@app.route('/paper_sell', methods=['POST'])
def paper_sell():
    return jsonify({"status": "success", "message": "Mock sell order"})

@app.route('/reset_paper_portfolio')
def reset_paper_portfolio():
    return jsonify({"status": "success", "message": "Mock reset"})

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
