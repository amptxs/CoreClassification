from flask import Flask, render_template, flash, request
from wtforms import Form, TextAreaField, validators, StringField, SubmitField
from wtforms.validators import InputRequired, Length
import predict

app = Flask(__name__)
app.config['SECRET_KEY'] = '420'


class ReusableForm(Form):
    name = TextAreaField('Описание:', validators=[InputRequired()])

    @app.route("/", methods=['GET', 'POST'])
    def send():
        form = ReusableForm(request.form)

        if request.method == 'POST':
            description = request.form['descriptionXGB']
            # Save the comment here.
            arr = predict.proba(description)
            for item in arr:
                flash(item)

        return render_template('descritpionClassification.html', form=form)


if __name__ == '__main__':
    app.run()
