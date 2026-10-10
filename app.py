import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Define the YieldPredictor class (must be included in the Streamlit app for standalone deployment)
class YieldPredictor:
    def __init__(self):
        self.model = None
        self.target_encoder = None
        self.one_hot_encoder = None
        self.X_train_columns = None
        # Store global medians for imputation as class attributes
        self.avg_dap = None
        self.avg_npsb = None
        self.avg_urea = None
        self.median_moisture = None
        self.median_micro = None
        self.median_half_moon = None
        self.global_median_area = None # Fallback for 'area'

    def load_assets(self, model_path, target_encoder_path, one_hot_encoder_path, X_train_columns_path, avg_dap, avg_npsb, avg_urea, median_moisture, median_micro, median_half_moon, global_median_area):
        """
        Loads the trained model and preprocessors from specified paths.
        Also stores the necessary imputation values and X_train_columns.
        """
        self.model = joblib.load(model_path)
        self.target_encoder = joblib.load(target_encoder_path)
        self.one_hot_encoder = joblib.load(one_hot_encoder_path)
        # Load X_train_columns from the saved path
        self.X_train_columns = joblib.load(X_train_columns_path)
        self.avg_dap = avg_dap
        self.avg_npsb = avg_npsb
        self.avg_urea = avg_urea
        self.median_moisture = median_moisture
        self.median_micro = median_micro
        self.median_half_moon = median_half_moon
        self.global_median_area = global_median_area
        st.success("Model and preprocessors loaded successfully!")

    def preprocess_input(self, raw_input_data: dict) -> pd.DataFrame:
        """
        Preprocesses a single raw input data point for model prediction.

        Args:
            raw_input_data (dict): A dictionary representing a single user input.

        Returns:
            pd.DataFrame: A preprocessed DataFrame ready for model prediction, with one row.
        """
        df_input = pd.DataFrame([raw_input_data])

        # Ensure all expected columns are present, fill with NaN if not provided
        expected_cols = ['Zone', 'Year', 'crop type (English)', 'irr round', 'DAP', 'NPSB', 'Urea', 'Moisture', 'Micro', 'Half Moon', 'area', 'productivity']
        for col in expected_cols:
            if col not in df_input.columns:
                df_input[col] = np.nan

        # Reorder columns to match the order used during X_raw creation
        df_input = df_input[expected_cols]

        # Imputation for numerical features (using stored medians)
        df_input['DAP'] = df_input['DAP'].fillna(self.avg_dap)
        df_input['NPSB'] = df_input['NPSB'].fillna(self.avg_npsb)
        df_input['Urea'] = df_input['Urea'].fillna(self.avg_urea)
        df_input['Moisture'] = df_input['Moisture'].fillna(self.median_moisture)
        df_input['Micro'] = df_input['Micro'].fillna(self.median_micro)
        df_input['Half Moon'] = df_input['Half Moon'].fillna(self.median_half_moon)
        df_input['area'] = df_input['area'].fillna(self.global_median_area)

        # Create a dummy 'yield' column for target encoding if not present
        if 'yield' not in df_input.columns:
            df_input['yield'] = 0  # Placeholder

        # Apply Target Encoding
        target_encode_cols = ['Zone', 'crop type (English)']
        df_input_encoded_target = self.target_encoder.transform(df_input[target_encode_cols])
        df_input = df_input.drop(columns=target_encode_cols)
        df_input = pd.concat([df_input, df_input_encoded_target], axis=1)

        # Apply One-Hot Encoding
        one_hot_encode_cols = ['irr round']
        encoded_irr_round = self.one_hot_encoder.transform(df_input[one_hot_encode_cols])
        one_hot_df_input = pd.DataFrame(encoded_irr_round, columns=self.one_hot_encoder.get_feature_names_out(one_hot_encode_cols), index=df_input.index)
        df_input = df_input.drop(columns=one_hot_encode_cols)
        df_input = pd.concat([df_input, one_hot_df_input], axis=1)

        # Remove 'productivity' column if it exists
        if 'productivity' in df_input.columns:
            df_input = df_input.drop(columns=['productivity'])

        # Align columns with X_train_columns - crucial for consistent model input
        processed_input = pd.DataFrame(columns=self.X_train_columns)
        for col in self.X_train_columns:
            if col in df_input.columns:
                processed_input[col] = df_input[col]
            else:
                processed_input[col] = 0.0 # Fill missing columns with 0 or appropriate default

        processed_input = processed_input[self.X_train_columns] # Ensure final order

        return processed_input

    def predict(self, raw_input_data: dict) -> float:
        """
        Takes raw input data, preprocesses it, and returns a yield prediction.
        """
        if self.model is None or self.target_encoder is None or self.one_hot_encoder is None:
            raise RuntimeError("Model or preprocessors not loaded. Call load_assets() first.")

        processed_input = self.preprocess_input(raw_input_data)
        prediction = self.model.predict(processed_input)[0]
        return prediction


# --- Streamlit Application ---
st.set_page_config(page_title="Crop Yield Prediction", layout="centered")
st.title("🌾 Crop Yield Prediction")
st.markdown("Enter the crop and environmental details to predict the yield.")

# File paths to the saved assets (adjust if needed)
# File paths to the saved assets (adjust if needed)
OPTIMIZED_MODEL_PATH = 'optimized_random_forest_regressor_model.pkl'
TARGET_ENCODER_PATH = 'target_encoder.pkl'
ONE_HOT_ENCODER_PATH = 'one_hot_encoder.pkl'
X_TRAIN_COLUMNS_PATH = 'X_train_columns.pkl'

@st.cache_resource
def get_and_load_predictor():
    predictor = YieldPredictor()
    predictor.load_assets(
        OPTIMIZED_MODEL_PATH,
        TARGET_ENCODER_PATH,
        ONE_HOT_ENCODER_PATH,
        X_TRAIN_COLUMNS_PATH, # Pass the path to X_train_columns
        # Imputation values from notebook kernel
        avg_dap=189407.10,
        avg_npsb=170353.75,
        avg_urea=137280.30,
        median_moisture=265507.0,
        median_micro=110394.0,
        median_half_moon=30696.0,
        global_median_area=108.0 # This was 'global_median' in your notebook
    )
    return predictor

yield_predictor = get_and_load_predictor()

# --- User Input Fields ---

st.header("Input Features")

# Get unique categories for dropdowns (hardcoded for simplicity in app, but should be dynamic)
# In a real application, these lists would be saved alongside the encoders or derived from training data.
example_zones = ['West Gojjam', 'East Gojjam', 'North Gojjam', 'Awi', 'Bahir Dar City', 'South Gonder', 'North Gonder', 'Central Gonder', 'West Gonder', 'North Wollo', 'South Wollo', 'North Shewa', 'Oromia', 'Wag Hemra', 'Weldiya', 'Debre Markos', 'Debre Tabor', 'Dese City', 'Gonder City', 'Kobo', 'Koga', 'Kombolcha', 'North West Gonder', 'Debre Birhan'] # Derived from `translation_map.values()`
example_crop_types = ['Teff', 'Wheat', 'Malt Barley', 'Food Barley', 'Maize', 'Oats', 'Sorghum', 'Finger Millet', 'Root Crops', 'F/Red Onion', 'Onion Seedlings', 'A/Red Onion', 'Garlic', 'Beetroot', 'Carrot', 'Sweet Potato', 'Potato', 'Vegetables', 'Lettuce', 'Swiss Chard', 'Cabbage', 'Tomato', 'Ethiopian Kale', 'Pepper', 'Watermelon', 'Spices', 'Chili Pepper', 'Fenugreek', 'Ginger', 'Rosemary', 'Coriander', 'Cardamom', 'Anise', 'Thyme', 'Turmeric', 'Black Cumin', 'White Cumin', 'Mustard', 'Fenugreek', 'Rue', 'Pulses', 'Chickpea', 'Lentil', 'Fodder Lentil', 'Ethiopian Pea', 'Faba Bean', 'Mung Bean', 'Common Bean', 'Pea', 'Soybean', 'Fruits', 'Mango', 'Avocado', 'Orange', 'Papaya', 'Lemon', 'Grape', 'Blueberry', 'Mandarin', 'Banana', 'Lime', 'Olive', 'Apple', 'Citron', 'Plum', 'Peach', 'Cashew', 'Guava', 'Pear', 'Cassava', 'Enset', 'Oilseeds', 'Sesame', 'Niger Seed', 'Flaxseed', 'Groundnut', 'Noug', 'Stimulant Crops', 'Chat', 'Tobacco', 'Sugarcane', 'Coffee', 'Gesho', 'Forage Crops', 'Fiber Crop', 'Cotton'] # Derived from `crop_type_translation_map.values()`

# Ensure these lists are unique and sorted for consistent dropdowns
example_zones = sorted(list(set(example_zones)))
example_crop_types = sorted(list(set(example_crop_types)))

selected_zone = st.selectbox("Zone", example_zones, index=example_zones.index('West Gojjam'))
selected_crop_type = st.selectbox("Crop Type", example_crop_types, index=example_crop_types.index('Teff'))
selected_irr_round = st.selectbox("Irrigation Round", ['1st', '2nd'], index=0)
year = st.slider("Year", min_value=2017, max_value=2025, value=2017)
area = st.number_input("Area (in Hectares)", min_value=0.0, value=100.0, format="%.2f")

st.subheader("Fertilizer Inputs")
dap = st.number_input("DAP (kg)", min_value=0.0, value=189407.10, format="%.2f")
npsb = st.number_input("NPSB (kg)", min_value=0.0, value=170353.75, format="%.2f")
urea = st.number_input("Urea (kg)", min_value=0.0, value=137280.30, format="%.2f")

st.subheader("Natural Resource Indicators")
moisture = st.number_input("Moisture", min_value=0.0, value=265507.0, format="%.2f")
micro = st.number_input("Micro", min_value=0.0, value=110394.0, format="%.2f")
half_moon = st.number_input("Half Moon", min_value=0.0, value=30696.0, format="%.2f")

# Create raw input dictionary
raw_input = {
    'Zone': selected_zone,
    'Year': year,
    'crop type (English)': selected_crop_type,
    'irr round': selected_irr_round,
    'DAP': dap,
    'NPSB': npsb,
    'Urea': urea,
    'Moisture': moisture,
    'Micro': micro,
    'Half Moon': half_moon,
    'area': area,
    'productivity': 0.0 # Placeholder, will be ignored/dropped by preprocess_input
}

if st.button("Predict Yield"):
    try:
        predicted_yield = yield_predictor.predict(raw_input)
        st.success(f"Predicted Yield: **{predicted_yield:.2f} Quintals**")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.json(raw_input) # Display input for debugging
