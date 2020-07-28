# Usage: python app.py
import os
import torch
import torch.nn as nn
import boto3
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, session
from werkzeug.utils import secure_filename
from fastai.vision import *
from fastai.metrics import error_rate
import numpy as np
import time
import uuid
import base64
from openpyxl import *
from model_retraining import retrain_model, insertImage

# connecting to the DynamoDb
dynamo_client = boto3.client('dynamodb')
DB = boto3.resource('dynamodb')
table = DB.Table('Utilisateur')

# loading the export.pkl file
learner = load_learner(path='')

# defenition of variables
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])


# convert the content of url to base64
def get_as_base64(url):
    return base64.b64encode(requests.get(url).content)


# function that predicts the proba and the type of img..............................................
def predict(file):
    x = open_image(file)  # Opening the image
    array = learner.predict(x)  # returning Tuple containing the category ++ the label ++ the prediction
    result = array[2].tolist()  # transforming the tensor to a list
    answer = np.argmax(result)  # Returns the indices of the maximum values along an axis.
    prob = max(result)  # returning the maximum probablity
    # depending on label of the answer we give the name of the bun
    if answer == 0:
        print("Label: Over")
    elif answer == 1:
        print("Label: Target")
    elif answer == 2:
        print("Label: Under")
    return answer, prob  # returning the answer and probability


# Returning a random string.........................................................................
def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4())  # Convert UUID format to a Python string.
    random = random.upper()  # Make all characters uppercase.
    random = random.replace("-", "")  # Remove the UUID '-'.
    return random[0:string_length]  # Return the random string.


# Allowing only Jpeg and jpg extentions....................................................................
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


# lanching the Flask app............................................................
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # The UPLOAD_FOLDER is where we will store the uploaded image files


# defining routes .....................................................................................


# rendering The template on the main page
@app.route("/")
def template_test():
    return render_template('template.html', label='', imagesource='../uploads/template.jpg')


# defining the type of methods
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        import time
        start_time = time.time()  # retuning the current time in seconds
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)  # secure filename before storing it directly on the filesystem.

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)  # adding the file to upload folder
            result, prob = predict(file_path)
            if result == 0:
                label = 'Over'
            elif result == 1:
                label = 'Target'
            elif result == 2:
                label = 'Under'

            print(result)
            print(file_path)

            filename = my_random_string(6) + filename  # adding a random string to file name

            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("--- %s seconds ---" % str(
                time.time() - start_time))  # printing the time that took to upload and giving the result
            return render_template('template.html', label=label, probabilty=prob, imagesource='../uploads/' + filename)


from flask import send_from_directory


# uploading the image to the uploads Folder.........
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


from werkzeug.middleware.shared_data import SharedDataMiddleware

app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads': app.config['UPLOAD_FOLDER']
})


# a function that Get the users from AWS..............
def get_users(index):
    response = table.get_item(
        Key={
            'Id': index
        }
    )
    return response["Item"]


# a function that inserts the Admin to AWS Database
def insert_admin(index, nom, prenom, email, password, IsSuperAdmin):
    response = table.put_item(
        Item={
            'Id': index,
            'Email': email,
            'IsSuperAdmin': IsSuperAdmin,
            'Nom': nom,
            'Password': password,
            'Prenom': prenom
        }
    )
    return response["ResponseMetadata"]["HTTPStatusCode"]


def update_admin(index, nom, prenom, email, password):
    response = table.update_item(
        Key={
            'Id': index
        },
        UpdateExpression="set Nom=:n, Prenom=:p, Email=:e , Password=:a",
        ExpressionAttributeValues={
            ':n': nom,
            ':p': prenom,
            ':e': email,
            ':a': password
        },
        ReturnValues="UPDATED_NEW"
    )
    return response


# Route for handling the login page logic.....................................
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        k = 1
        while k <= table.item_count + 1:
            if email == get_users(k).get('Email') and password == get_users(k).get('Password'):
                if get_users(k).get('IsSuperAdmin') == True:  # if the admin is a super admin than we redirect him to the superadmin page
                    session["logAdmin"] = True
                    session['username'] = get_users(k).get('Nom')
                    session['userprename'] = get_users(k).get('Prenom')
                    session['userEmail'] = get_users(k).get('Email')
                    session['userPassword'] = get_users(k).get('Password')
                    session['id'] = k
                    flash("your know logged in", "success")
                    return redirect(url_for('admin'))
                else:
                    session["log"] = True
                    session['username'] = get_users(k).get('Nom')
                    session['userprename'] = get_users(k).get('Prenom')
                    session['userEmail'] = get_users(k).get('Email')
                    session['userPassword'] = get_users(k).get('Password')
                    session['id'] = k
                    flash("your know logged in", "success")
                    return redirect(url_for('regular_admin'))
            else:
                k = k + 1
        flash("this account doesn't exists", "danger")
    return render_template('login.html')


# SuperAdmin page where he will be adding other admins.......................................
@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        nom = request.form['nom']
        prenom = request.form['prenom']
        isSuperAdmin = request.form['isSuperAdmin']
        confirmpassword = request.form['confirmpassword']
        if password == confirmpassword:
            insert_admin(table.item_count + 1, nom, prenom, email, password, isSuperAdmin)  # inserts the admin to DynamoDb Database
            flash("An admin has been added seccusfully", "success")
        else:
            flash("password does not match", "danger")
            return render_template('admin.html')
    return render_template('admin.html')


# regular admin page...........................................
@app.route('/regular_admin', methods=['GET', 'POST'])
def regular_admin():
    if request.method == 'POST':
        file = request.files['file']
        commentaire = request.form['Commentaire']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        shutil.move('../Bake_fastai_model_deploy/uploads/' + filename,
                    '../Bake_fastai_model_deploy/static/imageTesting/' + commentaire + '@' + filename)
        flash("Your image Has been added ", "primary")
    return render_template('regular_admin.html')


# the home page
@app.route('/home', methods=['GET', 'POST'])
def home():
    return render_template('home.html')


@app.route('/checkImages', methods=['GET', 'POST'])
def checkimages():
    image_names = os.listdir('../Bake_fastai_model_deploy/static/imageTesting/')  # List of images to be validated
    return render_template('checkImages.html', image_names=image_names)


# Deletes an image
@app.route('/Delete/<string:name>', methods=['GET', 'POST'])
def Delete(name):
    os.remove("../Bake_fastai_model_deploy/static/imageTesting/" + name)
    return redirect(url_for('checkimages'))


# Validates an image
@app.route('/Validate/<string:name>', methods=['GET', 'POST'])
def Validate(name):
    commentaire = name.split("@")[0]
    insertImage(name, commentaire)
    return redirect(url_for('checkimages'))


# Start the Training of the ML model
@app.route('/StartTraining')
def Train():
    retrain_model()
    return redirect(url_for('checkimages'))


@app.route('/Profil', methods=['GET', 'POST'])
def Profil():
    return render_template('Profil.html')


# Handling the logout
@app.route('/logout')
def log_out():
    session.clear()  # destroying the session
    flash("You are know logged out ", "success")
    return redirect(url_for('login'))


@app.route('/modifier', methods=['GET', 'POST'])
def modifier():
    if request.method == 'POST':
        nom = request.form['nom']
        prenom = request.form['prenom']
        password = request.form['password']
        email = request.form['email']
        confirmpassword = request.form['confirmpassword']
        index = session['id']
        if password == confirmpassword:
            update_admin(index, nom, prenom, email, password)
        else:
            flash("password doesn't match", "danger")
    return render_template('modifierProfil.html')


if __name__ == "__main__":
    app.secret_key = "1234567"
    app.debug = True
    app.run()
