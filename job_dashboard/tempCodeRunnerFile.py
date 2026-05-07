@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    role = data['role']
    location = data['location']

    prediction = model.predict(role, location)

    return jsonify({'salary': round(prediction, 2)})


if __name__ == '__main__':
    app.run(debug=True)