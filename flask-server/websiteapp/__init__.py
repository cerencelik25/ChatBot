from flask import Flask, session
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from lida import Manager, llm
from os import path, getenv
from flask_login import LoginManager, current_user
from msal import ConfidentialClientApplication
from flask_cors import CORS
from dotenv import load_dotenv
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize extensions
db = SQLAlchemy()
migrate = Migrate()
DB_NAME = "database.db"

def create_app():
    app = Flask(__name__)

    # Load configurations from environment variables
    app.config['SECRET_KEY'] = getenv('SECRET_KEY', 'default-secret-key')
    app.config['SQLALCHEMY_DATABASE_URI'] = getenv('SQLALCHEMY_DATABASE_URI', f'sqlite:///{DB_NAME}')
    app.config['UPLOAD_FOLDER'] = getenv('UPLOAD_FOLDER', 'uploads')
    app.config['GRAPH_FOLDER'] = getenv('GRAPH_FOLDER', 'graphs')
    app.config['IMAGE_FOLDER'] = getenv('IMAGE_FOLDER', 'generated_plots')

    # MSAL Configuration
    app.config['CLIENT_ID'] = getenv('CLIENT_ID')
    app.config['CLIENT_SECRET'] = getenv('CLIENT_SECRET')
    app.config['AUTHORITY'] = getenv('AUTHORITY')
    app.config['REDIRECT_URI'] = getenv('REDIRECT_URI')

    # OpenAI API Key
    app.config['OPENAI_API_KEY'] = getenv('OPENAI_API_KEY')
    app.config['JWT_SECRET_KEY'] = getenv('JWT_SECRET_KEY')

    # Initialize OpenAI client
    app.config['OPENAI_CLIENT'] = OpenAI(api_key=app.config['OPENAI_API_KEY'])
    
    # Initialize database and migration
    db.init_app(app)
    migrate.init_app(app, db)
    jwt = JWTManager(app)

    
    # Initialize LIDA Manager
    lida_manager = Manager(text_gen=llm("openai", api_key=app.config['OPENAI_API_KEY']))
    app.config['LIDA_MANAGER'] = lida_manager

    # Register blueprints
    from .views import views
    from .auth import auth
    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/auth')

    from .models import FileData, FileUpload, Question, User

    # Initialize login manager
    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        from .models import User
        return User.query.get(int(user_id))

    # Context processor to inject `current_user` into templates
    @app.context_processor
    def inject_user():
        return dict(user=current_user)

    
    CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}}, supports_credentials=True) # Enable CORS

    # Create the database if it doesn't exist
    create_database(app)

    return app

def create_database(app):
    db_path = path.join(app.root_path, DB_NAME)  # Ensure the correct path
    if not path.exists(db_path):
        with app.app_context():
            db.create_all()
            print(f"Database created at {db_path}")
