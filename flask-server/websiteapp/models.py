from . import db 
from flask_login import UserMixin
from sqlalchemy.sql import func

class FileUpload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    file_name = db.Column(db.String(150), nullable=False)
    file_type = db.Column(db.String(50), nullable=False)
    upload_date = db.Column(db.DateTime(timezone=True), default=func.now())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    first_name = db.Column(db.String(150))
    files = db.relationship('FileUpload', backref='user')

class FileData(db.Model):
    __tablename__ = 'file_data'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(255), nullable=False)
    file_data = db.Column(db.JSON, nullable=True)
    summary = db.Column(db.JSON, nullable=True)
    username = db.Column(db.String(255), nullable=False)
    questions = db.relationship('Question', backref='file_data', lazy=True)

class Question(db.Model):
    __tablename__ = 'question'
    id = db.Column(db.Integer, primary_key=True)
    question_text = db.Column(db.Text, nullable=False)
    file_data_id = db.Column(db.Integer, db.ForeignKey('file_data.id'), nullable=False)
    