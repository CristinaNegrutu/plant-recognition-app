from flask import Flask
from flask import render_template, flash, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import wikipediaapi

app = Flask(__name__)

UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.secret_key = "VERY_SECRET_KEY"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)

            return redirect(url_for('result', name=filename, path=path))


@app.route("/result")
def result():
    filename = request.args['name']
    path = request.args['path']
    name = filename.split(".")[0]
    wiki = wikipediaapi.Wikipedia('en')
    wiki_link = wiki.page(name)
    print("Page - Exists: %s" % wiki_link.exists())
    # solutie de moment pana legam aplicatia de retea
    if wiki_link.exists():
        wiki_title = wiki_link.title
        wiki_summary = wiki_link.summary
        has_result = True
    else:
        wiki_title = ""
        wiki_summary = ""
        has_result = False

    return render_template('result.html', name=filename, link=wiki_link.fullurl, title=wiki_title,
                           summary=wiki_summary, has_result=has_result)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
