from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Necessary for session management

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Input route
@app.route('/input', methods=['GET', 'POST'])
def input_data():
    if request.method == 'POST':
        input_text = request.form.get('input_text')
        file = request.files.get('input_file')
        if file:
            content = file.read().decode('utf-8')
            return redirect(url_for('select_algorithm', data=content))
        elif input_text:
            return redirect(url_for('select_algorithm', data=input_text))
        else:
            flash("Please provide input data")
    return render_template('input.html')

# Algorithm selection route
@app.route('/select_algorithm/<data>', methods=['GET', 'POST'])
def select_algorithm(data):
    if request.method == 'POST':
        algorithm = request.form.get('algorithm')
        if algorithm == 'premium_algorithm':
            return redirect(url_for('subscription'))
        else:
            prediction = f"Predicted using {algorithm}"  # Placeholder prediction
            return render_template('result.html', prediction=prediction)
    return render_template('select_algorithm.html', data=data)

# Subscription route
@app.route('/subscription')
def subscription():
    return render_template('subscription.html')

# Result route
@app.route('/result/<prediction>')
def result(prediction):
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
