from flask import Flask, render_template,request,flash,jsonify
from chatbot import *

jsonfile = intents['intents']
print("Outside")


app = Flask(__name__)
app.config['SECRET_KEY']='8ed6ybiweuftwued'

@app.route('/intent')
def intent_page():
    return render_template('intent.html',intents=jsonfile)


@app.route('/',methods=['GET','POST'])
def home_page():
    print("Inside Home page")
    message = ""
    output = ""
    prob = 0
    if request.method=="POST":
        if 'submit' in request.form:
            message = request.form.get('name')
            ints = predict_class(message)
            output,prob = get_response(ints,intents)
        else:
            message = ""
            output = ""
            prob=0
        return render_template("home.html",
                                message=message,
                                output=output + " ( Prob: " + str(round(prob*100,2)) + " % )")

    else:
        return render_template("home.html",output=output)


if __name__ == '__main__':
    app.run(debug=True)
