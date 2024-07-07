from flask import Flask, request, session, flash, render_template, redirect, jsonify
from config import  MY_DOMAIN, MY_PORT, STRIPE_PRICE_ID, TEST_STRIPE_SECRET_KEY
import re
import db
import json
import uuid
import stripe
import os

stripe.api_key = TEST_STRIPE_SECRET_KEY

app = Flask(__name__, static_url_path='/static', static_folder='static')



def get_header(**kwargs):
    """
    Returns the rendered template for the header/navbar.

    Args:
        **kwargs: Keyword arguments to pass to the template.

    Returns:
        str: Rendered template for the header/navbar.
    """
    return render_template("navbar.html", **kwargs)


@app.route("/success", methods=['GET'])
def success():
    """
    Renders the success.html template with the header/navbar.

    Returns:
        str: Rendered template for the success page.
    """
    return render_template("success.html", header=get_header())

@app.route("/cancel", methods=['GET'])
def cancel():
    """
    Renders the cancel.html template with the header/navbar.

    Returns:
        str: Rendered template for the cancel page.
    """
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
    try:
        checkout_session = stripe.checkout.Session.create(
            line_items=[
                {
                    # Provide the exact Price ID (for example, pr_1234) of the product you want to sell
                    'price': 'price_1PZNsiRqCSLozqidx1A5ez2N',
                    'quantity': 1,
                },
            ],
            mode='payment',
            success_url=MY_DOMAIN + '/success.html',
            cancel_url=MY_DOMAIN + '/cancel.html',
        )
    except Exception as e:
        return str(e)

    return redirect(checkout_session.url, code=303)

@app.route("/login", methods = ["GET"])
def get_login():
    """
    Renders the login.html template with the header/navbar.

    Returns:
        str: Rendered template for the login page.
    """
    return render_template("login.html", header=get_header())


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