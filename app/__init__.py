# app/__init__.py

import os
from flask import Flask
from flask_cors import CORS  # <-- IMPORT THIS
import joblib
import pandas as pd

def create_app():
    app = Flask(__name__)
    CORS(app)  # <-- ADD THIS LINE TO ENABLE CORS

    # --- Configuration ---
    # (The rest of the file stays exactly the same)
    # ...
