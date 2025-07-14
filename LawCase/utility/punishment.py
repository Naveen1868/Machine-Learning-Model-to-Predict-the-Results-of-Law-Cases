def punish(request):
    return render(request, 'users/training.html')

import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from django.shortcuts import render
from django.conf import settings
import os


path = os.path.join(settings.MEDIA_ROOT, 'FIR_DATASET.csv')
data = pd.read_csv(path)

# Encode categorical features
lb = LabelEncoder()
data['Description'] = lb.fit_transform(data['Description'])
data['Bailable'] = lb.fit_transform(data['Bailable'])
data['Offense'] = lb.fit_transform(data['Offense'])
data['Punishment'] = lb.fit_transform(data['Punishment'])
data['Cognizable'] = lb.fit_transform(data['Cognizable'])
data['Court'] = lb.fit_transform(data['Court'])

# Define features and labels
x = data[['Bailable', 'Offense', 'Cognizable', 'Court']]  # Four features for the model
y = data[['Punishment']]

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=35)

# Train the RandomForest model
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

# Evaluate the model
pre = rf.predict(x_test)
acc = accuracy_score(y_test, pre)
print(f'Accuracy: {acc}')

# Punishment labels
labels = [
    'Death', 'Imprisonment for Life', '10 Years + Fine', '7 Years + Fine', '3 Years + Fine',
    '2 Years + Fine', '1 Year + Fine', 'Simple Imprisonment for 3 Years + Fine',
    'Simple Imprisonment for 2 Years + Fine', 'Simple Imprisonment for 1 Year + Fine',
    'Simple Imprisonment for 6 Months + Fine', 'Simple Imprisonment for 3 Months + Fine',
    'Simple Imprisonment for 1 Month + Fine', 'Fine', '3 Months or Fine or Both',
    '6 Months or Fine or Both', '2 Years or Fine or Both', '3 Years or Fine or Both',
    '1 Year or Fine or Both', '7 Years or Fine or Both', 'Rigorous Imprisonment for 10 Years + Fine',
    'Rigorous Imprisonment for 7 Years + Fine', 'Rigorous Imprisonment for not less than 7 Years',
    'Rigorous Imprisonment for 20 years to Imprisonment for Natural-Life + Fine paid to the victim',
    'Rigorous Imprisonment for 20 years to Imprisonment for Natural-Life or Death',
    'Rigorous Imprisonment for 5 to 10 years + Fine', 'Imprisonment for not less than 7 Years, but up to Life',
    'Imprisonment for Life or Rigorous Imprisonment for 10 Years + Fine',
    'Imprisonment for Life or 10 Years or Rigorous Imprisonment for 10 Years + Fine',
    'Death or Imprisonment for Life + Fine', 'Death or Imprisonment for Life or 10 Years + Fine',
    'First conviction 2 Years + Fine, then 5 Years + Fine', 'First conviction 3 Years + Fine, then 7 Years + Fine',
    '5 Years or Fine or Both', '10 years to Life + Fine paid to the victim', '5 to 7 years + Fine',
    '2 to 7 years + Fine', '3 to 7 years + Fine', 'Up to 3 years or Fine or Both',
    'Up to 3 years + Fine for first conviction', '1 to 5 years + Fine'
]

# Prediction view
def predict_punishment(request):
    if request.method == 'POST':
        # Fetch form data (assuming it's numeric input now or converted properly in your form handling)
        bailable = int(request.POST.get('bailable'))
        offense = int(request.POST.get('Offense'))
        cognizable = int(request.POST.get('cognizable'))
        court = int(request.POST.get('court'))
        
        print(f"Bailable: {bailable}, Offense: {offense}, Cognizable: {cognizable}, Court: {court}")
        
        # Prepare input data for prediction
        input_data = [[bailable, offense, cognizable, court]]
        
        # Make prediction using the RandomForest model
        predicted_index = rf.predict(input_data)[0]
        print(f"Predicted index: {predicted_index}")
        if predicted_index <=39:
            op=predicted_index
        else:
            op=random.randint(0,41)
        predicted_punishment = labels[op]
        
        print(f"Predicted punishment: {predicted_punishment}")

        # Return the result to the template
        return render(request, 'users/output_punish.html', {'predicted_punishment': predicted_punishment})

    return render(request, 'users/punishment.html')


