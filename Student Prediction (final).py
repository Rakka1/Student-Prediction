import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import gradio as gr
from sklearn.metrics import r2_score

# Load and preprocess data
data = pd.read_csv('student-por.csv')

# Drop columns that are not useful or leak info
exclude_columns = ['school','traveltime','nursery','higher','famrel','freetime',
                   'goout','Pstatus','reason','address','famsize',
                   'romantic','Dalc','Walc','paid']
data = data.drop(columns=exclude_columns)

# Separate features and target
X = data.drop(['G3'], axis=1)
y_G3 = data['G3']

# Drop rows with missing values
df = pd.concat([X, y_G3], axis=1).dropna()
X = df.drop(['G3'], axis=1)
y_G3 = df['G3']

# Encode categorical columns only
label_encoders = {}
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Train RandomForestRegressor
X_train, X_test, y_train, y_test = train_test_split(
    X, y_G3, test_size=0.2, random_state=42
)

model_G3 = RandomForestRegressor(random_state=42)
model_G3.fit(X_train, y_train)

# Evaluate model
train_pred = model_G3.predict(X_train)
test_pred = model_G3.predict(X_test)
train_score = r2_score(y_train, train_pred)
test_score = r2_score(y_test, test_pred)
print(f"G3 Model - Train Score: {train_score:.2f}, Test Score: {test_score:.2f}")

#Prediction function with safe handling
def predict_G3(*inputs):
    input_dict = dict(zip(X.columns, inputs))
    input_df = pd.DataFrame([input_dict])

    try:
        # Transform categorical columns safely
        for col in label_encoders:
            val = input_df[col][0]
            if val not in label_encoders[col].classes_:
                return f"Invalid input for {col}. Must be one of: {list(label_encoders[col].classes_)}"
            input_df[col] = label_encoders[col].transform([val])

        prediction = model_G3.predict(input_df)[0]
        return f"Predicted Final Grade (G3): {round(prediction,2)}"

    except Exception as e:
        return f"Prediction Error: {str(e)}"

# Gradio UI
with gr.Blocks() as app:
    gr.Markdown("# Student Final Grade Prediction")
    gr.Markdown("Enter student details to predict the final grade (G3).")

    with gr.Row():

        with gr.Column():
            gr.Markdown("### Student Information")
            sex = gr.Dropdown(choices=sorted(data['sex'].astype(str).unique()), label="Sex")
            age = gr.Slider(15, 22, step=1, label="Age")

            guardian = gr.Dropdown(choices=sorted(data['guardian'].astype(str).unique()), label="Guardian")

            gr.Markdown("### Family Background")
            Medu = gr.Slider(0, 4, step=1, label="Mother Education")
            Fedu = gr.Slider(0, 4, step=1, label="Father Education")
            Mjob = gr.Dropdown(choices=sorted(data['Mjob'].astype(str).unique()), label="Mother Job")
            Fjob = gr.Dropdown(choices=sorted(data['Fjob'].astype(str).unique()), label="Father Job")
            famsup = gr.Dropdown(choices=sorted(data['famsup'].astype(str).unique()), label="Family Support")

        with gr.Column():
            gr.Markdown("### Academic Factors")
            studytime = gr.Slider(1, 4, step=1, label="Study Time")
            failures = gr.Slider(0, 3, step=1, label="Past Failures")
            schoolsup = gr.Dropdown(choices=sorted(data['schoolsup'].astype(str).unique()), label="School Support")
            activities = gr.Dropdown(choices=sorted(data['activities'].astype(str).unique()), label="Activities")

            gr.Markdown("### Previous Grades")
            G1 = gr.Slider(1, 100, step=1, label="Previous Grade 1")
            G2 = gr.Slider(1, 100, step=1, label="Previous Grade 2")
            
            gr.Markdown("### Health & Other")
            health = gr.Slider(1, 5, step=1, label="Health Status")
            internet = gr.Dropdown(choices=sorted(data['internet'].astype(str).unique()), label="Internet Access")
            absences = gr.Number(label="Absences", minimum=0, maximum=93)


    # inputs
    inputs = [sex, age, Medu, Fedu, Mjob, Fjob,
    guardian, studytime, failures, schoolsup, famsup,
    activities, internet, health, absences, G1, G2]

    predict_btn = gr.Button("Predict Final Grade")
    result = gr.Textbox(label="Prediction Result")

    predict_btn.click(fn=predict_G3, inputs=inputs, outputs=result)



app.launch()
print(X.columns.tolist())