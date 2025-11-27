import streamlit
import numpy
import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error,r2_score

data = pandas.read_csv('energy_efficiency_data.csv')
df = pandas.DataFrame(data)

X = df.iloc[:,:-2]
Y = df.iloc[:,8:]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr = LinearRegression()
model = lr.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

print("R2 Score:",r2_score(Y_test,Y_pred))
print("MAPE:",mean_absolute_percentage_error(Y_test,Y_pred))

streamlit.header("Energy Efficiency Prediction")

def input_features():

    x1 = streamlit.number_input("Relative Compactness",0.0,1.0,0.5)
    x2 = streamlit.number_input("Surface Area",300.0,900.0,600.0)
    x3 = streamlit.number_input("Wall Area",200.0,500.0,300.0)
    x4 = streamlit.number_input("Roof Area",100.0,400.0,200.0)
    x5 = streamlit.number_input("Overall Height",3.0,8.0,5.0)
    x6 = streamlit.number_input("Orientation",2,6,4)
    x7 = streamlit.number_input("Glazing Area",0.0,0.4,0.2)
    x8 = streamlit.number_input("Glazing Area Distribution",0,5,3)

    data = {
        "Relative_Compactness" : x1,
        "Surface_Area" : x2,
        "Wall_Area" : x3,
        "Roof_Area" : x4,
        "Overall_Height" : x5,
        "Orientation" : x6,
        "Glazing_Area" : x7,
        "Glazing_Area_Distribution" : x8
    }

    input_df = pandas.DataFrame(data,index=[0])
    return input_df

features = input_features()

scaled_features = scaler.transform(features)

result = model.predict(scaled_features)

if streamlit.button("Predict"):
    streamlit.subheader("The Predicted values are:")
    streamlit.write("Heating Load:",result[0][0])
    streamlit.write("Cooling Load:",result[0][1])