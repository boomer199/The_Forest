from flask import Flask, request, session, flash, render_template, redirect, jsonify
from config import MONGO_URL
from werkzeug.security import generate_password_hash, check_password_hash
from bson.objectid import ObjectId
import stripe
import os   
from pymongo.mongo_client import MongoClient
from threading import Thread
from models import multistock 

#get env vars
MY_DOMAIN = os.environ["MY_DOMAIN"]
MY_PORT = os.environ["MY_PORT"]
STRIPE_PRICE_ID = os.environ["STRIPE_PRICE_ID"]
TEST_STRIPE_SECRET_KEY = os.environ["TEST_STRIPE_SECRET_KEY"]

stripe.api_key = TEST_STRIPE_SECRET_KEY

app = Flask(__name__, static_url_path='/static', static_folder='static')

app.secret_key = os.urandom(24)

client = MongoClient(MONGO_URL)

db = client["forestdb"]

users = db.users

# MongoDB setup for Flask-Session
app.config["SESSION_TYPE"] = "mongodb"
app.config["SESSION_PERMANENT"] = True
app.config["PERMANENT_SESSION_LIFETIME"] = 86400  # Sessions last for one day
app.config["SESSION_MONGODB"] = MongoClient(MONGO_URL)
app.config["SESSION_MONGODB_DB"] = "forestdb"
app.config["SESSION_MONGODB_COLLECT"] = "sessions"



def get_header():
    """
    Returns the rendered template for the header/navbar.

    Returns:
        str: Rendered template for the header/navbar.
    """
    user = get_user()
    return render_template("navbar.html", user=user)


def get_user():
    """
    Retrieves the user details from MongoDB.

    Returns:
        dict: User details including account type, username, and email.
    """
    if is_logged_in():
        user_id = session.get('user_id')
        user = users.find_one({'_id': ObjectId(user_id)})
        if user:
            return {
                'account_type': user.get('premium', False),
                'username': user.get('username', ''),
                'email': user.get('email', '')
            }
    return None



def is_logged_in():
    return 'user_id' in session

def process_premium_data(timeframe, smoothing, custom_stocks):
    data = multistock.run_script()
    global premium_data
    premium_data = data
    
    

@app.route("/premium", methods=['POST'])
def premium_post():
    if not is_logged_in():
        return render_template('login.html', header=get_header())
    
    user_id = session.get('user_id')
    user = users.find_one({'_id': ObjectId(user_id)})
    if not user or not user.get('premium', False):
        return render_template('no_access.html', header=get_header())

    try:
        form_data = request.form
        print("Received form data:", form_data)  # Debug print
        
        timeframe = form_data.get('timeframe')
        smoothing = form_data.get('smoothing') == 'true'  # Checkbox returns 'true' as string if checked
        custom_stocks = form_data.get('customStocks')

        missing_fields = []
        if not timeframe:
            missing_fields.append('timeframe')
        if custom_stocks is None:  # Allow empty string for custom stocks
            missing_fields.append('customStocks')

        if missing_fields:
            error_message = f"Missing required form fields: {', '.join(missing_fields)}"
            app.logger.error(f"Error in premium_post: {error_message}")
            return jsonify({"error": error_message}), 400

        # Start processing data in a background thread
        Thread(target=process_premium_data, args=(timeframe, smoothing, custom_stocks)).start()

        # Immediately return the loading page
        return render_template("premium_load.html", header=get_header())
    except Exception as e:
        app.logger.error(f"Error in premium_post: {str(e)}")
        return jsonify({"error": "An error occurred processing your request"}), 500

@app.route("/premium_data", methods=['GET'])
def get_premium_data():
    if not is_logged_in():
        return jsonify({"error": "Not logged in"}), 401
    
    user_id = session.get('user_id')
    user = users.find_one({'_id': ObjectId(user_id)})
    if not user or not user.get('premium', False):
        return jsonify({"error": "No premium access"}), 403

    global premium_data
    if premium_data is not None:
        return jsonify(premium_data)
    else:
        return jsonify({"status": "processing"}), 202

@app.route('/premium', methods=['GET'])
def premium():
    if not is_logged_in():
        return render_template('login.html', header=get_header())
    user_id = session.get('user_id')
    user = users.find_one({'_id': ObjectId(user_id)})
    if user and user.get('premium', False):
        d = multistock.run_script()
        return render_template('premium.html', header=get_header(), data = d)
    
    return render_template('no_access.html', header=get_header())

    
    
    
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        username = request.form['username']


        # Check if the user already exists
        if users.find_one({'email': email}):
            return jsonify({"error": "email already exists"}), 409

        # Hash the password and insert a new user
        hashed_pwd = generate_password_hash(password)
        users.insert_one({
            'password': hashed_pwd,
            'email': email,
            'username': username,
            'premium': False
        })

        return render_template('login.html', header = get_header())
    return render_template('login.html', header = get_header())

@app.route('/logout')
def logout():
    session.clear()
    return render_template('login.html', header=get_header())


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        user = users.find_one({'email': email})
        if user and check_password_hash(user['password'], password):
            session['user_id'] = str(user['_id'])  # Store the user ID in the session
            session.modified = True
            print("1")
            return render_template('index.html', header=get_header())
        else:
            print("2")
            return render_template('login.html', header=get_header(), error="Invalid credentials")
    return render_template('login.html', header=get_header())



from stripe.error import StripeError

@app.route("/success")
def success():
    session_id = request.args.get('session_id')
    if not session_id:
        return jsonify({'error': 'Session ID is missing.'}), 400

    try:
        stripe_session = stripe.checkout.Session.retrieve(session_id)
        if stripe_session.payment_status == 'paid':
            user_id = stripe_session.metadata['user_id']
            users.update_one({'_id': ObjectId(user_id)}, {'$set': {'premium': True}})
            return render_template("success.html", header=get_header())
    except StripeError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'An error occurred, unable to update premium status.'}), 500

    render_template("cancel.html", header=get_header())

@app.route("/cancel.html", methods=['GET'])
def cancel_html():
    """
    An alternative route to serve the cancel.html template.

    Returns:
        str: Rendered template for the cancel page.
    """
    if not is_logged_in():
        return render_template('login.html', header=get_header())
    else:
        return render_template("cancel.html", header=get_header())



@app.route("/demo", methods=["POST"])
def demo_post():
    user_id = session.get('user_id')
    user = users.find_one({'_id': ObjectId(user_id)})
    flash("Please purchase premium to utilize these features.", "Denial")
    d = multistock.run_script()
    if user and user.get('premium', True):
        return render_template('premium.html', header=get_header())
    return render_template("demo.html", header=get_header(), data=d)

@app.route("/demo", methods=["GET"])
def demo():
    user_id = session.get('user_id')
    user = users.find_one({'_id': ObjectId(user_id)})
    """
    Renders the demo.html template with the header/navbar.

    Returns:
        str: Rendered template for the demo page.
    """
    d = multistock.run_script()
    if user and user.get('premium', True):
        return render_template('premium.html', header=get_header())
    return render_template("demo.html", header=get_header(), data=d)

@app.route("/checkout", methods=['GET'])
def checkout():
    """
    Renders the checkout.html template with the header/navbar.

    Returns:
        str: Rendered template for the checkout page.
    """
    return render_template("checkout.html", header=get_header())

@app.route('/create-checkout-session', methods=['POST'])
def create_checkout_session():
    if not is_logged_in():
        return render_template('login.html', header=get_header())
    
    user_id = session.get('user_id')
    try:
        checkout_session = stripe.checkout.Session.create(
            line_items=[
                {
                    'price': STRIPE_PRICE_ID,
                    'quantity': 1,
                },
            ],
            mode='payment',
            metadata={'user_id': user_id},
            success_url=MY_DOMAIN + '/success?session_id={CHECKOUT_SESSION_ID}',
            cancel_url=MY_DOMAIN + '/cancel.html',
        )
    except Exception as e:
        return str(e)

    return redirect(checkout_session.url, code=303)

@app.route("/demo_load", methods=["GET"])
def demo_load():
    return render_template("demo_load.html", header=get_header())


@app.route("/premium_load", methods=["GET"])
def premium_load():
    return render_template("premium_load.html", header=get_header())

@app.route("/")
@app.route("/index")
def home():
    """
    Renders the index.html template with the header/navbar.

    Returns:
        str: Rendered template for the home page.
    """
    return render_template("index.html", header=get_header())

if __name__ == '__main__':
    app.run(port=MY_PORT)
    