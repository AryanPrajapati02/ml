import streamlit as st
import pandas as pd
import pickle
import compress_pickle as cp


try:
    import compress_pickle as cp
except ImportError:
    st.write("Installing compress-pickle...")
    st.code("pip install compress-pickle==2.1.0", language="bash")
    get_ipython().system('pip install compress-pickle==2.1.0')
    import compress_pickle as cp

# Now load your model
with open('car_model.pkl.gz', 'rb') as f:
    model = cp.load(f, compression="gzip")
car_name_encoder = pickle.load(open('car_name_encoder.pkl', 'rb'))
company_encoder = pickle.load(open('company_encoder.pkl', 'rb'))
fuel_encoder = pickle.load(open('fuel_encoder.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
# Load the dataset

df = pd.read_csv('data.csv')


st.title("Car Prediction App")
st.write("""It is a Old car price detector where you have to provide some details and this
          model will tell you the price of the given input.""")

st.subheader("Lets Predict")

cp = st.selectbox("**Company**",df["company"].unique(),index = None)

if cp:
    st.write("You Selected: ",cp)
    
""
if cp:
     filtered_models = df[df["car_name"].str.startswith(cp)]["car_name"].unique()
    
else:
    filtered_models = df["car_name"].unique()     
    
car_name = st.selectbox("**Model**" , filtered_models , index=None)


if car_name:
  st.write("You Selected: ",car_name)



km = st.number_input("**Kilometers Driven**", 0.0, None, step = 1000.0)

if km:
    st.write("You Entered: ",km)


year = st.slider("**Year**",min_value = 0,max_value=60,value = 0,step = 1)

manufactureyear = 2024-year
st.write("**Manufacture Year**",manufactureyear)
""

ft = st.radio("**Fuel Type**",["Petrol","Diesel"],index = None)
if ft:
    st.write(ft)
""

def pipe (new_input):
  new_input['car_name'] = car_name_encoder.transform(new_input['car_name'])
  new_input['company'] = company_encoder.transform(new_input['company'])
  new_input['fuel_type'] = fuel_encoder.transform(new_input['fuel_type'])
  new_data = pd.DataFrame(scaler.transform(new_input), columns=new_input.columns )
  ans = model.predict(new_data)

  return ans[0]
if st.button("**Price Predictor**"):

    input_data = pd.DataFrame({
        'car_name': [car_name],
        'company': [cp],
        'year': [year],
        'kms_driven': [km],
        'fuel_type': [ft]
    })
    p= pipe(input_data)
    st.write(f'Estimated Car Price: {p:,.2f}rs')





