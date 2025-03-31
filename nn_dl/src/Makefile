# Makefile - for local dev and deployment

# Run local dev server (with Flask debug mode)
run:
	python run.py

# Run with Gunicorn (production-ready WSGI server)
serve:
	gunicorn -w 4 -b 0.0.0.0:5000 wsgi:app

# Run local with Flask environment loaded
dev:
	FLASK_ENV=development python run.py
