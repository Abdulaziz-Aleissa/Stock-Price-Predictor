"""
Modular Stock Price Predictor Application

This module now serves as the entry point for the refactored modular application.
The original monolithic code has been extracted into separate services, models, and routes.
"""

from .application import create_app

# Create the Flask application using the factory pattern
app = create_app()

if __name__ == '__main__':
    app.run(debug=True)