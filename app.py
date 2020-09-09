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
from model_retraining import retrain_model_royal, retrain_model_BM,retrain_model_regular,insertImage_Regular,insertImage_BM,insertImage_royal
from flask_mail import Mail,Message
from botocore.config import Config
from PIL import Image



# lanching the Flask app............................................................
app = Flask(__name__)


mail=Mail(app)
#S3 bucket
s3 = boto3.resource('s3')
BUCKET = "bun-image-profile"
# connecting to the DynamoDb
DB = boto3.resource('dynamodb')
table = DB.Table('Utilisateur')
path=''
# loading the export.pkl file
learner_BM = load_learner(path,'model_bm.pkl')
learner_Reg = load_learner(path,'model_Regular.pkl')
learner_Roy = load_learner(path,'model_royal.pkl')



# defenition of variables
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # The UPLOAD_FOLDER is where we will store the uploaded image files

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config["MAIL_PORT"] = 465
app.config['MAIL_USERNAME'] = 'Buns.vision@gmail.com'
app.config['MAIL_PASSWORD'] = 'bigdata@2020'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
# convert the content of url to base64
def get_as_base64(url):
    return base64.b64encode(requests.get(url).content)


# function that predicts the proba and the type of img..............................................
def predict_Reg(file):
    x = open_image(file)  # Opening the image
    #x.resize(torch.Size([x.shape[0],1280,960]))
    array = learner_Reg.predict(x)  # returning Tuple containing the category ++ the label ++ the prediction
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

def predict_BM(file):
    x = open_image(file)  # Opening the image
    array = learner_BM.predict(x)  # returning Tuple containing the category ++ the label ++ the prediction
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

def predict_Royal(file):
    x = open_image(file)  # Opening the image
    array = learner_Roy.predict(x)  # returning Tuple containing the category ++ the label ++ the prediction
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




# defining routes .....................................................................................
@app.route("/identify")
def identify():
    return render_template('identify.html')

# rendering The template on the main page
@app.route('/regular')
def regular():
    return render_template('template.html', label='', imagesource='../uploads/template.jpeg')


# defining the type of methods
@app.route('/regular', methods=['GET', 'POST'])
def upload_file_regular():
    if request.method == 'POST':
        import time
        start_time = time.time()  # retuning the current time in seconds
        file = request.files['fileup']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)  # secure filename before storing it directly on the filesystem.
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)  # adding the file to upload folder
            result, prob = predict_Reg(file_path)
            proba=round(prob,4)*100
            if result == 0:
                label = 'Over'
            elif result == 1:
                label = 'Target'
            elif result == 2:
                label = 'Under'

            print(result)
            print(file_path)
            print(proba)

            filename = my_random_string(6) + filename  # adding a random string to file name

            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("--- %s seconds ---" % str(
                time.time() - start_time))  # printing the time that took to upload and giving the result
            return render_template('template.html', label=label, probabilty=proba, imagesource='../uploads/' + filename)

@app.route('/BM')
def BM():
    return render_template('template.html', label='', imagesource='../uploads/template.jpeg')


# defining the type of methods
@app.route('/BM', methods=['GET', 'POST'])
def upload_file_BM():
    if request.method == 'POST':
        import time
        start_time = time.time()  # retuning the current time in seconds
        file = request.files['fileup']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)  # secure filename before storing it directly on the filesystem.

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)  # adding the file to upload folder
            result, prob = predict_BM(file_path)
            proba=round(prob,4)*100
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
            return render_template('template.html', label=label, probabilty=proba, imagesource='../uploads/' + filename)


@app.route('/royal')
def royal():
    return render_template('template.html', label='', imagesource='../uploads/template.jpeg')


# defining the type of methods
@app.route('/royal', methods=['GET', 'POST'])
def upload_file_royal():
    if request.method == 'POST':
        import time
        start_time = time.time()  # retuning the current time in seconds
        file = request.files['fileup']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)  # secure filename before storing it directly on the filesystem.
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)  # adding the file to upload folder
            result, prob = predict_Royal(file_path)
            proba=round(prob,4)*100
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
            return render_template('template.html', label=label, probabilty=proba, imagesource='../uploads/' + filename)

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
            'Photo':"https://cdn.business2community.com/wp-content/uploads/2017/08/blank-profile-picture-973460_640.png",
            'Prenom': prenom
        }
    )
    return response["ResponseMetadata"]["HTTPStatusCode"]


def update_admin(index, nom, prenom, email, password,photo):
    response = table.update_item(
        Key={
            'Id': index
        },
        UpdateExpression="set Nom=:n, Prenom=:p, Email=:e , Password=:a,Photo=:l ",
        ExpressionAttributeValues={
            ':n': nom,
            ':p': prenom,
            ':e': email,
            ':a': password,
            ':l': photo
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
        while k <= table.item_count:
            if email == get_users(k).get('Email') and password == get_users(k).get('Password'):
                if get_users(k).get('IsSuperAdmin') == True:  # if the admin is a super admin than we redirect him to the superadmin page
                    session["logAdmin"] = True
                    session['username'] = get_users(k).get('Nom')
                    session['photo']=get_users(k).get('Photo')
                    session['userprename'] = get_users(k).get('Prenom')
                    session['userEmail'] = get_users(k).get('Email')
                    session['userPassword'] = get_users(k).get('Password')
                    session['id'] = k
                    flash("your know logged in", "success")
                    return redirect(url_for('admin'))
                else :
                    session["log"] = True
                    session['username'] = get_users(k).get('Nom')
                    session['photo']=get_users(k).get('Photo')
                    session['userprename'] = get_users(k).get('Prenom')
                    session['userEmail'] = get_users(k).get('Email')
                    session['userPassword'] = get_users(k).get('Password')
                    session['id'] = k
                    flash("your know logged in", "success")
                    return redirect(url_for('regular_admin'))
            else:
                k = k + 1
        flash("the password or email are icorrect", "danger")
        return redirect(url_for('login'))
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
        file = request.files['fileup']
        commentaire = request.form['Commentaire']
        Type=request.form['type']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        shutil.move('../Bake_fastai_model_deploy/uploads/' + filename,
                    '../Bake_fastai_model_deploy/static/imageTesting/' + commentaire + '@' + Type + '@' + filename)
        flash("Your image Has been added ", "primary")
    return render_template('regular_admin.html')


# the home page
@app.route('/', methods=['GET', 'POST'])
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
    Type=name.split("@")[1]
    if Type=="regular":
        insertImage_Regular(name, commentaire)
    if Type=="royal":
        insertImage_royal(name,commentaire)
    if Type=="BM":
        insertImage_BM(name,commentaire)
    return redirect(url_for('checkimages'))


# Start the Training of the ML model
@app.route('/StartTrainingRegular')
def TrainReg():
    retrain_model_regular()
    message=Message("Updating The Regular Model",sender="Buns.vision@gmail.com",recipients=[session['userEmail']])
    message.body = "The Machine learning Model of the regular bun is Updated successfully"
    mail.send(message)
    flash("The machine learning model is  Updated","primary")
    return redirect(url_for('identify'))


@app.route('/StartTrainingBM')
def TrainBM():
    retrain_model_BM()
    message=Message("Updating The Regular Model",sender="Buns.vision@gmail.com",recipients=[session['userEmail']])
    message.body = "The Machine learning Model of the Big Mac bun is Updated successfully"
    mail.send(message)
    flash("The machine learning model is Updated","primary")
    return redirect(url_for('identify'))


@app.route('/StartTrainingRoyal')
def TrainRoy():
    retrain_model_royal()
    message=Message("Updating The Regular Model",sender="Buns.vision@gmail.com",recipients=[session['userEmail']])
    message.body = "The Machine learning Model of the royal bun is Updated successfully"
    mail.send(message)
    flash("The machine learning model is Updated","primary")
    return redirect(url_for('identify'))


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
        url_photo="https://cdn.business2community.com/wp-content/uploads/2017/08/blank-profile-picture-973460_640.png"
        file = request.files['file']
        nom = request.form['nom']
        prenom = request.form['prenom']
        password_0 = request.form['password_0']
        password = request.form['password']
        email = request.form['email']
        confirmpassword = request.form['confirmpassword']
        index = session['id']
        if file.filename=="":
            if password_0 == session['userPassword']:
                if password == confirmpassword:
                    update_admin(index, nom, prenom, email, password,url_photo)
                else :
                    flash("password doesn't match", "danger")
            else :
                flash("wrong password ", "danger")
        else :
            file_path = os.path.join(app.config['UPLOAD_FOLDER'],file.filename)
            file.save(file_path)
            s3.Bucket(BUCKET).upload_file("./uploads/"+file.filename, "images/"+str(index)+file.filename)
            os.remove("./uploads/"+file.filename)
            if password_0 == session['userPassword']:
                if password == confirmpassword:
                    update_admin(index, nom, prenom, email, password,"https://bun-image-profile.s3.us-east-2.amazonaws.com/images/"+str(index)+file.filename)
                else :
                     flash("password doesn't match", "danger")
            else :
                flash("wrong password ", "danger") 
        return redirect(url_for('log_out'))
    return render_template('modifierProfil.html')

if __name__ == "__main__":
    app.secret_key = "1234567"
    app.debug = True
    app.run(host='0.0.0.0')
