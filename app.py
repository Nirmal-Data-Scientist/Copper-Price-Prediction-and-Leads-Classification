from streamlit_extras.add_vertical_space import add_vertical_space
import streamlit as st
import pandas as pd
import joblib

data = pd.read_csv('Datasets and Models/data.csv')

cols_mapping = {
    'customer_id': 'customer_id',
    'item_type': 'item_type',
    'application': 'application',
    'country_code': ['country_code_' + str(col) for col in data['country_code'].unique() if col != 107],
    'product_ref': ['product_ref_' + str(col) for col in data['product_ref'].unique() if col != 1282007633],
    'leads': ['leads_' + str(col) for col in data['leads'].unique()]
}

@st.cache_resource
def load_models():
    target_encoder_reg = joblib.load(r'Datasets and Models/te_reg.pkl')
    target_encoder_cls = joblib.load(r'Datasets and Models/te_cls.pkl')
    scaler_reg = joblib.load(r'Datasets and Models/scaler_reg.pkl')
    scaler_cls = joblib.load(r'Datasets and Models/scaler_cls.pkl')
    pca_reg = joblib.load(r'Datasets and Models/pca_reg.pkl')
    pca_cls = joblib.load(r'Datasets and Models/pca_cls.pkl')
    reg_model = joblib.load(r'Datasets and Models/reg_model.pkl')
    cls_model = joblib.load(r'Datasets and Models/cls_model.pkl')
    return target_encoder_reg, target_encoder_cls, scaler_reg, scaler_cls, pca_reg, pca_cls, reg_model, cls_model

def preprocess_input_reg(target_encoder, scaler, pca, input_data):

    df = input_data.copy()

    # Initialize the one-hot encoded columns with 0
    df_ohe_country_code = pd.DataFrame(0, index=df.index, columns=cols_mapping['country_code'])
    df_ohe_product_ref = pd.DataFrame(0, index=df.index, columns=cols_mapping['product_ref'])

    # Set 1 for the specified values in the input data
    user_country_code = 'country_code_' + str(df['country_code'].iloc[0])
    user_product_ref = 'product_ref_' + str(df['product_ref'].iloc[0])
    user_leads = df['leads'].iloc[0]

    df_ohe_country_code[user_country_code] = 1
    df_ohe_product_ref[user_product_ref] = 1

    # Update leads value in the original dataframe
    df['leads'] = 1 if user_leads == 'Won' else 0

    df = pd.concat([df, df_ohe_country_code, df_ohe_product_ref], axis=1)

    df.drop(['country_code', 'product_ref'], axis=1, inplace=True)
    
    desired_column_order = ['quantity_tons', 'customer_id', 'item_type', 'application', 'thickness',
                            'width', 'leads', 'country_code_113', 'country_code_25',
                            'country_code_26', 'country_code_27', 'country_code_28',
                            'country_code_30', 'country_code_32', 'country_code_38',
                            'country_code_39', 'country_code_40', 'country_code_77',
                            'country_code_78', 'country_code_79', 'country_code_80',
                            'country_code_84', 'country_code_89', 'product_ref_1332077137',
                            'product_ref_164141591', 'product_ref_164336407',
                            'product_ref_164337175', 'product_ref_1665572032',
                            'product_ref_1665572374', 'product_ref_1665584320',
                            'product_ref_1665584642', 'product_ref_1668701376',
                            'product_ref_1668701698', 'product_ref_1668701718',
                            'product_ref_1668701725', 'product_ref_1670798778',
                            'product_ref_1671863738', 'product_ref_1671876026',
                            'product_ref_1690738206', 'product_ref_1690738219',
                            'product_ref_1693867550', 'product_ref_1693867563',
                            'product_ref_1721130331', 'product_ref_1722207579',
                            'product_ref_611728', 'product_ref_611733', 'product_ref_611993',
                            'product_ref_628112', 'product_ref_628117', 'product_ref_628377',
                            'product_ref_640400', 'product_ref_640405', 'product_ref_640665',
                            'product_ref_929423819']

    df = df[desired_column_order]
    
    df = target_encoder.transform(df)

    continuous_vars = ['quantity_tons', 'thickness', 'width']
    df[continuous_vars] = scaler.transform(df[continuous_vars])

    df_pca = pca.transform(df)

    return df_pca

def preprocess_input_cls(target_encoder, scaler, pca, input_data):
    
    df = input_data.copy()

    # Initialize the one-hot encoded columns with 0
    df_ohe_country_code = pd.DataFrame(0, index=df.index, columns=cols_mapping['country_code'])
    df_ohe_product_ref = pd.DataFrame(0, index=df.index, columns=cols_mapping['product_ref'])

    # Set 1 for the specified values in the input data
    user_country_code = 'country_code_' + str(df['country_code'].iloc[0])
    user_product_ref = 'product_ref_' + str(df['product_ref'].iloc[0])

    df_ohe_country_code[user_country_code] = 1
    df_ohe_product_ref[user_product_ref] = 1

    df = pd.concat([df, df_ohe_country_code, df_ohe_product_ref], axis=1)

    df.drop(['country_code', 'product_ref'], axis=1, inplace=True)
    
    desired_column_order = ['quantity_tons', 'customer_id', 'item_type', 'application', 'thickness',
                            'width', 'selling_price', 'country_code_113', 'country_code_25',
                            'country_code_26', 'country_code_27', 'country_code_28',
                            'country_code_30', 'country_code_32', 'country_code_38',
                            'country_code_39', 'country_code_40', 'country_code_77',
                            'country_code_78', 'country_code_79', 'country_code_80',
                            'country_code_84', 'country_code_89', 'product_ref_1332077137',
                            'product_ref_164141591', 'product_ref_164336407',
                            'product_ref_164337175', 'product_ref_1665572032',
                            'product_ref_1665572374', 'product_ref_1665584320',
                            'product_ref_1665584642', 'product_ref_1668701376',
                            'product_ref_1668701698', 'product_ref_1668701718',
                            'product_ref_1668701725', 'product_ref_1670798778',
                            'product_ref_1671863738', 'product_ref_1671876026',
                            'product_ref_1690738206', 'product_ref_1690738219',
                            'product_ref_1693867550', 'product_ref_1693867563',
                            'product_ref_1721130331', 'product_ref_1722207579',
                            'product_ref_611728', 'product_ref_611733', 'product_ref_611993',
                            'product_ref_628112', 'product_ref_628117', 'product_ref_628377',
                            'product_ref_640400', 'product_ref_640405', 'product_ref_640665',
                            'product_ref_929423819']

    df = df[desired_column_order]
    
    df = target_encoder.transform(df)

    continuous_vars = ['quantity_tons', 'thickness', 'width', 'selling_price']
    df[continuous_vars] = scaler.transform(df[continuous_vars])

    df_pca = pca.transform(df)

    return df_pca

def main():
    
    st.set_page_config(page_title="Industrial Copper Modelling", page_icon=r"Related Images and Videos/copper.png", layout="wide")
    
    st.title("Copper Modelling App")
    
    add_vertical_space(2)

    target_encoder_reg, target_encoder_cls, scaler_reg, scaler_cls, pca_reg, pca_cls, reg_model, cls_model = load_models()

    st.subheader("Copper Price Prediction")
    
    add_vertical_space(2)
    
    col1_reg, col2_reg, col3_reg = st.columns(3)

    with col1_reg:
        quantity_tons_reg = st.number_input("Quantity in tons", value=139.322857, min_value=0.0)
        add_vertical_space(1)
        leads_reg = st.selectbox("Leads", data['leads'].unique(), index=0)
        add_vertical_space(1)
        thickness_reg = st.number_input("Thickness", value=0.80, min_value=0.0)
        add_vertical_space(2)
        
    with col2_reg:
        customer_id_reg = st.selectbox("Customer ID", data['customer_id'].unique(), index=0)
        add_vertical_space(1)
        item_type_reg = st.selectbox("Item Type", data['item_type'].unique(), index=0)
        add_vertical_space(1)
        width_reg = st.number_input("Width", value=1210.0, min_value=0.0)
        add_vertical_space(2)
        
    with col3_reg:
        country_code_reg = st.selectbox("Country Code", [val for val in data['country_code'].unique() if val != 107], index=0)
        add_vertical_space(1)
        application_reg = st.selectbox("Application", data['application'].unique(), index=0)
        add_vertical_space(1)
        product_ref_reg = st.selectbox("Product Reference", [val for val in data['product_ref'].unique() if val != 1282007633], index=0)
        add_vertical_space(2)
        
    input_data_reg = pd.DataFrame({
        'quantity_tons': [quantity_tons_reg],
        'customer_id': [customer_id_reg],
        'country_code': [country_code_reg],
        'leads': [leads_reg],
        'item_type': [item_type_reg],
        'application': [application_reg],
        'thickness': [thickness_reg],
        'width': [width_reg],
        'product_ref': [product_ref_reg]
    })
    
    if st.button("Predict Price"):
        
        input_data_reg_pca = preprocess_input_reg(target_encoder_reg, scaler_reg, pca_reg, input_data_reg)        
        
        prediction_reg = reg_model.predict(input_data_reg_pca)
        st.write("Copper Price:", prediction_reg[0])

    add_vertical_space(2)
    
    st.subheader("Leads Classification")
    
    add_vertical_space(2)
    
    col1_cls, col2_cls, col3_cls = st.columns(3)

    with col1_cls:
        quantity_tons_cls = st.number_input("Quantity in tons", value=139.322857, min_value=0.0, key=1)
        add_vertical_space(1)
        selling_price_cls = st.number_input("Selling Price", value=1047.00, min_value=0.0, key=2)
        add_vertical_space(1)
        thickness_cls = st.number_input("Thickness", value=0.80, min_value=0.0, key=3)
        add_vertical_space(2)
        
    with col2_cls:
        customer_id_cls = st.selectbox("Customer ID", data['customer_id'].unique(), index=0, key=4)
        add_vertical_space(1)
        item_type_cls = st.selectbox("Item Type", data['item_type'].unique(), index=0, key=5)
        add_vertical_space(1)
        width_cls = st.number_input("Width", value=1210.0, min_value=0.0, key=6)
        add_vertical_space(2)
    with col3_cls:
        country_code_cls = st.selectbox("Country Code", [val for val in data['country_code'].unique() if val != 107], index=0, key=7)
        add_vertical_space(1)
        application_cls = st.selectbox("Application", data['application'].unique(), index=0, key=8)
        add_vertical_space(1)
        product_ref_cls = st.selectbox("Product Reference", [val for val in data['product_ref'].unique() if val != 1282007633], index=0, key=9)
        add_vertical_space(2)

    input_data_cls = pd.DataFrame({
        'quantity_tons': [quantity_tons_cls],
        'customer_id': [customer_id_cls],
        'country_code': [country_code_cls],
        'item_type': [item_type_cls],
        'application': [application_cls],
        'thickness': [thickness_cls],
        'width': [width_cls],
        'product_ref': [product_ref_cls],
        'selling_price': [selling_price_cls]
    })
    
    
    if st.button("Predict Leads"):
    
        input_data_cls_pca = preprocess_input_cls(target_encoder_cls, scaler_cls, pca_cls, input_data_cls)
    
        prediction_cls = cls_model.predict(input_data_cls_pca)
        
        if prediction_cls[0] == 1:
            st.write("Leads: Won")
        else:
            st.write("Leads: Lost")
    
    sb = st.sidebar
    sb.image(r"https://www.naturalfoodseries.com/wp-content/uploads/2017/08/Copper.jpg", width=250)
    sb.title("Copper Modelling")
    sb.write("""
             This app utilizes Random Forest Regressor and Random Forest Classifier models to predict copper prices and classify leads as Won or Lost based on given input data respectively.
             Interact with the user-friendly interface to make accurate predictions for your industrial copper related scenarios.
             """)
    
if __name__ == "__main__":
    main()