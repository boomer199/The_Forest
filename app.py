from flask import Flask, request, session, flash, render_template, redirect, jsonify

from config import MY_DOMAIN, MY_PORT, MY_REDIS_PORT, MY_REDIS_HOST
from os import getenv
import redis
import re
import db
import json
import uuid




app = Flask(__name__)



def get_header(**kwargs):
    return render_template("navbar.html", **kwargs)

@app.route("/")
@app.route("/index")
def home():
     return render_template("index.html", header=get_header())


if __name__ == '__main__':
    app.run(port=MY_PORT)