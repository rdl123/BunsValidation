#Usage: python app.py
import os
 
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from fastai.vision import *
from fastai.metrics import error_rate
import numpy as np
import time
import uuid
import base64


#loading the export file 
learner = load_learner(path='') 

#defenition of variables
UPLOAD_FOLDER = 'uploads' 
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg']) 


#convert the content of url to base64
def get_as_base64(url):
    return base64.b64encode(requests.get(url).content)


#function that predicts the proba and the type of img..............................................
def predict(file):
    x = open_image(file)  #Opening the image
    array = learner.predict(x) #returning Tuple containing the category ++ the label ++ the prediction
    result = array[2].tolist() #transforming the tensor to a list
    answer = np.argmax(result)  # Returns the indices of the maximum values along an axis.
    prob = max(result) #returning the maximum probablity
    #depending on label of the answer we give the name of the bun
    if answer == 0:
        print("Label: Over")
    elif answer == 1:
        print("Label: Target")
    elif answer == 2:
        print("Label: Under")
    return answer, prob  #returning the answer and probability


#Returning a random string.........................................................................
def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.


#Allowing only Jpeg and jpg extentions....................................................................
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

#lanching the Flask app............................................................
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER #The UPLOAD_FOLDER is where we will store the uploaded image files


#defining routes .....................................................................................


#rendering The template on the main page
@app.route("/")
def template_test():
    return render_template('template.html', label='', imagesource='../uploads/template.jpg')

# defining the type of methods
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        import time
        start_time = time.time() #retuning the current time in seconds
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)  #secure filename before storing it directly on the filesystem.

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path) #adding the file to upload folder
            result , prob = predict(file_path)
            if result == 0:
                label = 'Over'
            elif result == 1:
                label = 'Target'          
            elif result == 2:
                label = 'Under'

            print(result)
            print(file_path)

            filename = my_random_string(6) + filename # adding a random string to file name

            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("--- %s seconds ---" % str (time.time() - start_time))   #printing the time that took to upload and giving the result
            return render_template('template.html', label=label , probabilty = prob , imagesource='../uploads/' + filename)

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

from werkzeug.middleware.shared_data  import SharedDataMiddleware
app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  app.config['UPLOAD_FOLDER']
})

if __name__ == "__main__":
    app.debug=False
    app.run(  )