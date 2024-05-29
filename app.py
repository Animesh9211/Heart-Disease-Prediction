from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

app = Flask(__name__)

# Load and preprocess the dataset
df = pd.read_csv('D:\\Dataset\\heart_disease_data.csv')
target_column = 'target'
categorical_columns = df.select_dtypes(include=['object']).columns
numerical_columns = df.select_dtypes(include=['number']).columns.drop(target_column)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ])

X = df.drop(target_column, axis=1)
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1)

input_columns = X.columns

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [x for x in request.form.values()]
    input_data = np.array(input_data, dtype=str).reshape(1, -1)
    if input_data.shape[1] != len(input_columns):
        return render_template('index.html', prediction_text=f'Expected {len(input_columns)} values but got {input_data.shape[1]}. Please try again.')
    input_df = pd.DataFrame(input_data, columns=input_columns)
    input_preprocessed = preprocessor.transform(input_df)
    prediction = model.predict(input_preprocessed)
    output = 'Heart Disease' if prediction > 0.5 else 'No Heart Disease'
    return render_template('index.html', prediction_text=f'The predicted class for the new patient data is: {output}')

if __name__ == "__main__":
    app.run(debug=True)
