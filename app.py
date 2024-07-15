from flask import Flask, request, session, flash, render_template, redirect, jsonify
from config import  MY_DOMAIN, MY_PORT, STRIPE_PRICE_ID, TEST_STRIPE_SECRET_KEY, MONGO_URI
from werkzeug.security import generate_password_hash, check_password_hash
from bson.objectid import ObjectId
import stripe
import os   
from pymongo.mongo_client import MongoClient

stripe.api_key = TEST_STRIPE_SECRET_KEY

app = Flask(__name__, static_url_path='/static', static_folder='static')

app.secret_key = os.urandom(24)

client = MongoClient(MONGO_URI)

db = client["forestdb"]

users = db.users

# MongoDB setup for Flask-Session
app.config["SESSION_TYPE"] = "mongodb"
app.config["SESSION_PERMANENT"] = True
app.config["PERMANENT_SESSION_LIFETIME"] = 86400  # Sessions last for one day
app.config["SESSION_MONGODB"] = MongoClient(MONGO_URI)
app.config["SESSION_MONGODB_DB"] = "forestdb"
app.config["SESSION_MONGODB_COLLECT"] = "sessions"


def get_header(**kwargs):
    """
    Returns the rendered template for the header/navbar.

    Args:
        **kwargs: Keyword arguments to pass to the template.

    Returns:
        str: Rendered template for the header/navbar.
    """
    return render_template("navbar.html", **kwargs)

def is_logged_in():
    return 'user_id' in session

@app.route('/premium', methods=['GET'])
def premium():
    if not is_logged_in():
        return render_template('login.html', header=get_header())
    user_id = session.get('user_id')
    user = users.find_one({'_id': ObjectId(user_id)})
    if user and user.get('premium', False):
        return render_template('premium.html', header=get_header())
    return render_template('not_premium.html', header=get_header(), message="Please purchase a premium plan to access this page.")

    
    
    
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
            return render_template("success.html", header=get_header(), message="Thank you for your purchase!")
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
    