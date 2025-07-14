import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from django.conf import settings
from django.shortcuts import render

# Load dataset
data_path = settings.MEDIA_ROOT + "//" + 'FIR_DATASET.csv'
data = pd.read_csv(data_path)
# data = data.drop(['URL'], axis=1)

# Convert all columns to strings before encoding
data['Description'] = data['Description'].astype(str)
data['Bailable'] = data['Bailable'].astype(str)
data['Offense'] = data['Offense'].astype(str)
data['Punishment'] = data['Punishment'].astype(str)
data['Cognizable'] = data['Cognizable'].astype(str)
data['Court'] = data['Court'].astype(str)

# Label encoding
lb = LabelEncoder()
data['Description'] = lb.fit_transform(data['Description'])
data['Bailable'] = lb.fit_transform(data['Bailable'])
data['Offense'] = lb.fit_transform(data['Offense'])
data['Punishment'] = lb.fit_transform(data['Punishment'])
data['Cognizable'] = lb.fit_transform(data['Cognizable'])
data['Court'] = lb.fit_transform(data['Court'])

x = data[['Description', 'Offense', 'Punishment', 'Cognizable', 'Court']]
y = data['Bailable']

x_train, x_test, y_train, y_test = train_test_split(x, y.ravel(), test_size=0.2, random_state=31)

# Convert all columns to strings for TfidfVectorizer
x_train_str = x_train.applymap(str)
x_test_str = x_test.applymap(str)

# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit and transform x_train
x_train_vec = vectorizer.fit_transform(x_train_str.apply(lambda x: ' '.join(x), axis=1))

# Transform x_test
x_test_vec = vectorizer.transform(x_test_str.apply(lambda x: ' '.join(x), axis=1))

# Initialize RandomForestClassifier
rf = RandomForestClassifier()

# Fit the RandomForestClassifier
rf.fit(x_train_vec, y_train)

# Predict and evaluate
y_pred = rf.predict(x_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')


def bailable(request):
    y_pre = rf.predict(vectorizer.transform(x_test.apply(lambda x: ' '.join(x.map(str)), axis=1)))
    accuracy = accuracy_score(y_pre, y_test)
    print(f'Accuracy: {accuracy}')
    return render(request, "users/accuracy.html", {'accuracy': accuracy})

def pred_bail(request):
    if request.method == 'POST':
        Description = request.POST.get('Description', '')
        Offense = request.POST.get('Offense', '')
        Punishment = request.POST.get('Punishment', '')
        Cognizable = request.POST.get('Cognizable', '')
        Court = request.POST.get('Court', '')

        # Vectorize the input data
        input_data_text = ' '.join([Description, Offense, Punishment, Cognizable, Court])
        # print(input_data_text)
        input_data_vec = vectorizer.transform([input_data_text])
        # print(input_data_vec)

        pred = rf.predict(input_data_vec)[0]
        print(pred)

        if pred == 0:
            pred = 'Bailable'
        elif pred == 1:
            pred = 'Non-Bailable'

        return render(request, 'users/output.html', {'pred': pred})

    return render(request, 'users/prediction.html', {})
