from flask import Flask, render_template, request, jsonify
from pydantic import ValidationError

from api.schemas.hello_schema import HelloRequestModel
from service.hello_service import HelloService
from helpers.utils import format_response


def create_flask_app(config, template_dir=None, static_dir=None):
    app = Flask(__name__, static_folder=static_dir or 'static', template_folder=template_dir or 'templates')
    app.config.update(config)
    app.state = type('State', (), {'config': config})()

    hello_service = HelloService(config)

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/hello', methods=['GET'])
    def hello_get():
        try:
            result = hello_service.say_hello()
            return jsonify(format_response("Hello GET", result))
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/hello', methods=['POST'])
    def hello_post():
        try:
            data = request.get_json(force=True)
            validated_data = HelloRequestModel(**data)
            result = hello_service.process_data(validated_data.dict())
            return jsonify(format_response("Hello POST", result))
        except ValidationError as ve:
            return jsonify({"error": ve.errors()}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app
