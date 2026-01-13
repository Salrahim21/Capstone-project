import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import folium
from streamlit_folium import folium_static
import sys
import glob
import os
import warnings
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Optional, List
warnings.filterwarnings('ignore')

# ====================== CUSTOM TRANSFORMER DEFINITIONS ======================
# Define flexible custom transformers that can handle different initialization patterns

class UniversalBinaryMapper:
    """Universal BinaryMapper that handles any initialization pattern and ensures 2D output"""
    def __init__(self, *args, **kwargs):
        # Handle all possible initialization patterns
        self.args = args
        self.kwargs = kwargs
        
        # Common attributes that might be expected
        self.mapping = kwargs.get('mapping', {})
        self.default_value = kwargs.get('default_value', 0)
        self.columns = kwargs.get('columns', None)
        
        # Handle positional arguments
        if len(args) > 0:
            if isinstance(args[0], dict):
                self.mapping = args[0]
            elif isinstance(args[0], (list, tuple)):
                self.columns = args[0]
        
        if len(args) > 1:
            if isinstance(args[1], (int, float)):
                self.default_value = args[1]
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Ensure input is DataFrame
        if not isinstance(X, pd.DataFrame):
            if hasattr(X, 'shape') and len(X.shape) == 1:
                X = pd.DataFrame(X.reshape(1, -1))
            else:
                X = pd.DataFrame(X)
        
        X_transformed = X.copy()
        
        # Apply binary mapping logic
        for column in X_transformed.columns:
            if column in self.mapping:
                X_transformed[column] = X_transformed[column].map(self.mapping).fillna(self.default_value)
            elif X_transformed[column].dtype == 'bool':
                X_transformed[column] = X_transformed[column].astype(int)
            elif X_transformed[column].dtype == 'object':
                # Convert categorical to numeric if possible
                try:
                    # Try to convert to numeric first
                    X_transformed[column] = pd.to_numeric(X_transformed[column], errors='ignore')
                    if X_transformed[column].dtype == 'object':
                        # If still object, do label encoding
                        unique_vals = X_transformed[column].unique()
                        mapping = {val: i for i, val in enumerate(unique_vals)}
                        X_transformed[column] = X_transformed[column].map(mapping).fillna(0)
                except:
                    pass
        
        # Ensure output is 2D
        result = X_transformed.values if isinstance(X_transformed, pd.DataFrame) else X_transformed
        if len(result.shape) == 1:
            result = result.reshape(1, -1)
        
        return result
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
    
    def get_params(self, deep=True):
        """Required for sklearn compatibility"""
        return {'mapping': self.mapping, 'default_value': self.default_value}
    
    def set_params(self, **params):
        """Required for sklearn compatibility"""
        for key, value in params.items():
            setattr(self, key, value)
        return self

class UniversalCategoricalEncoder:
    """Universal categorical encoder that ensures 2D output"""
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.categories = kwargs.get('categories', {})
        self.handle_unknown = kwargs.get('handle_unknown', 'ignore')
        self.columns = kwargs.get('columns', None)
        
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        for column in X.select_dtypes(include=['object', 'category']).columns:
            unique_vals = X[column].unique()
            self.categories[column] = {val: i for i, val in enumerate(unique_vals)}
        return self
    
    def transform(self, X):
        # Ensure input is DataFrame
        if not isinstance(X, pd.DataFrame):
            if hasattr(X, 'shape') and len(X.shape) == 1:
                X = pd.DataFrame(X.reshape(1, -1))
            else:
                X = pd.DataFrame(X)
        
        X_transformed = X.copy()
        
        # Apply categorical encoding
        for column in X_transformed.columns:
            if X_transformed[column].dtype in ['object', 'category']:
                if column in self.categories:
                    X_transformed[column] = X_transformed[column].map(self.categories[column]).fillna(0)
                else:
                    # Fallback encoding
                    unique_vals = X_transformed[column].unique()
                    mapping = {val: i for i, val in enumerate(unique_vals)}
                    X_transformed[column] = X_transformed[column].map(mapping).fillna(0)
        
        # Ensure output is 2D
        result = X_transformed.values if isinstance(X_transformed, pd.DataFrame) else X_transformed
        if len(result.shape) == 1:
            result = result.reshape(1, -1)
            
        return result
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
    
    def get_params(self, deep=True):
        return {'categories': self.categories, 'handle_unknown': self.handle_unknown}
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

class UniversalPriceNormalizer:
    """Universal price normalizer that ensures 2D output"""
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.scale = kwargs.get('scale', 1.0)
        self.method = kwargs.get('method', 'standard')
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Ensure input is DataFrame
        if not isinstance(X, pd.DataFrame):
            if hasattr(X, 'shape') and len(X.shape) == 1:
                X = pd.DataFrame(X.reshape(1, -1))
            else:
                X = pd.DataFrame(X)
        
        X_transformed = X.copy()
        
        # Apply normalization if needed
        for column in X_transformed.columns:
            if 'price' in str(column).lower() and X_transformed[column].dtype in ['float64', 'int64']:
                if self.scale != 1.0:
                    X_transformed[column] = X_transformed[column] / self.scale
        
        # Ensure output is 2D
        result = X_transformed.values if isinstance(X_transformed, pd.DataFrame) else X_transformed
        if len(result.shape) == 1:
            result = result.reshape(1, -1)
            
        return result
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
    
    def get_params(self, deep=True):
        return {'scale': self.scale, 'method': self.method}
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

# First, register universal transformers in multiple locations
universal_classes = {
    'BinaryMapper': UniversalBinaryMapper,
    'binary_mapper': UniversalBinaryMapper,
    'binary_mapper_occ': UniversalBinaryMapper,
    'binary_mapper_price': UniversalBinaryMapper,
    'BinaryMapperOcc': UniversalBinaryMapper,
    'BinaryMapperPrice': UniversalBinaryMapper,
    'CategoricalEncoder': UniversalCategoricalEncoder,
    'PriceNormalizer': UniversalPriceNormalizer,
}

# Register in multiple namespaces to ensure compatibility
for name, cls in universal_classes.items():
    # Register in main module
    sys.modules['__main__'].__dict__[name] = cls
    globals()[name] = cls
    
    # Register in builtins for global access
    import builtins
    setattr(builtins, name, cls)
    
    # Register in custom_transformers namespace if it exists
    try:
        import custom_transformers
        setattr(custom_transformers, name, cls)
    except:
        pass

# Create a fake custom_transformers module if it doesn't exist
if 'custom_transformers' not in sys.modules:
    import types
    fake_module = types.ModuleType('custom_transformers')
    for name, cls in universal_classes.items():
        setattr(fake_module, name, cls)
    sys.modules['custom_transformers'] = fake_module

custom_transformer_status = "✅ Universal transformers loaded"

# Try to import original transformers, but use universals as fallback
try:
    from custom_transformers import *
    # Replace with universal versions anyway to ensure compatibility
    for name, cls in universal_classes.items():
        globals()[name] = cls
        sys.modules['custom_transformers'].__dict__[name] = cls
    custom_transformer_status = "✅ Original transformers replaced with universal versions"
except ImportError:
    custom_transformer_status = "✅ Using universal transformers (original not found)"
# Define the custom function exactly as it was when the pipeline was saved
def map_binary_t_f(X):
    return X.replace({'t': 1, 'f': 0})

# Set page config
st.set_page_config(
    page_title="Airbnb Cape Town Analysis", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏖️"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF5A5F, #00A699);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF5A5F;
        margin: 0.5rem 0;
    }
    .status-success {
        color: #28a745;
    }
    .status-error {
        color: #dc3545;
    }
    .prediction-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)
# ====================== FEATURE DEFINITION ======================
# ===== PRICE feature groups =====
numeric_features_price = [
    'accommodates',
    'bathrooms',
    'bedrooms',
    'beds',
    'avg_rating',
    'number_of_reviews',
    'calculated_host_listings_count',
    'host_response_rate'
]

binary_features_price = ['host_is_superhost']

categorical_features_price = [
    'neighbourhood_cleansed',
    'property_type'
]

all_features_price = (
    numeric_features_price + binary_features_price + categorical_features_price
)


# ===== OCCUPANCY feature groups =====
numeric_features_occupancy = [
    'accommodates',
    'bathrooms',
    'bedrooms',
    'beds',
    'avg_rating',
    'calculated_host_listings_count',
    'host_response_rate',
    'avg_price',
    'minimum_nights',
    'availability_rate',
    'days_since_last_review',
    'number_of_reviews'
]

binary_features_occupancy = ['host_is_superhost']

categorical_features_occupancy = [
    'neighbourhood_cleansed',
    'property_type'
]

all_features_occupancy = (
    numeric_features_occupancy
    + binary_features_occupancy
    + categorical_features_occupancy
)


# ====================== TRAINING DATA LOADING ======================
@st.cache_data
def load_training_features():
    """Load the training feature CSV files to understand feature structure"""
    feature_data = {
        'price_features': pd.DataFrame(),
        'occupancy_features': pd.DataFrame(),
    }
    
    feature_file_status = {}
    
    try:
        feature_data['price_features'] = pd.read_csv('app/train_price_data.csv')
        # Remove target column to get just features
        if 'price' in feature_data['price_features'].columns:
            feature_data['price_features'] = feature_data['price_features'].drop(columns=['price'])
        feature_file_status['price_features'] = f'✅ Loaded ({feature_data["price_features"].shape})'
    except Exception as e:
        feature_file_status['price_features'] = f'❌ Error: {str(e)[:50]}'
    
    try:
        feature_data['occupancy_features'] = pd.read_csv('app/train_occupancy_data.csv')
        # Remove target column to get just features  
        if 'occupancy' in feature_data['occupancy_features'].columns:
            feature_data['occupancy_features'] = feature_data['occupancy_features'].drop(columns=['occupancy'])
        feature_file_status['occupancy_features'] = f'✅ Loaded ({feature_data["occupancy_features"].shape})'
    except Exception as e:
        feature_file_status['occupancy_features'] = f'❌ Error: {str(e)[:50]}'
    
    return feature_data, feature_file_status

# Load the training features to understand structure
feature_data, feature_file_status = load_training_features()

# Default values - now derived from training data if available
def get_default_values():
    defaults = {
        'property_types': ['Apartment', 'House', 'Villa', 'Condominium', 'Townhouse'],
        'room_types': ['Entire home/apt', 'Private room', 'Shared room'],
        'neighbourhoods': ['Ward 115', 'Ward 1', 'Ward 10', 'City Bowl', 'Sea Point']
    }
    
    # Update from training data if available
    if not feature_data['price_features'].empty:
        df = feature_data['price_features']
        if 'property_type' in df.columns:
            defaults['property_types'] = sorted(df['property_type'].unique().tolist())
        if 'room_type' in df.columns:
            defaults['room_types'] = sorted(df['room_type'].unique().tolist())
        if 'neighbourhood_cleansed' in df.columns:
            defaults['neighbourhoods'] = sorted(df['neighbourhood_cleansed'].unique().tolist())
        elif 'neighbourhood' in df.columns:
            defaults['neighbourhoods'] = sorted(df['neighbourhood'].unique().tolist())
    
    return defaults

DEFAULT_VALUES = get_default_values()

# ====================== MODEL LOADING (ORIGINAL APPROACH) ======================
@st.cache_resource
def load_models():
    models = {'price': None, 'occupancy': None}
    loading_status = {'price': 'Not loaded', 'occupancy': 'Not loaded'}

    # --- Price model loading ---
        # --- Price model loading ---
    file_path = "app/price_prediction_model_v2.pkl"  # Explicit file name

    if os.path.exists(file_path):
        try:
            models['price'] = joblib.load(file_path)  # Always use joblib for consistency
            # Validate the loaded object
            from sklearn.base import BaseEstimator
            if not isinstance(models['price'], BaseEstimator):
                st.error(f"❌ Loaded price model is not a valid estimator (type: {type(models['price'])}).")
                st.stop()
            loading_status['price'] = f"✅ Loaded ({file_path})"
        except Exception as e:
            st.error(f"❌ Failed to load price model from {file_path}: {str(e)}")
            st.stop()
    else:
        st.error(f"❌ Price model file not found: {file_path}")
        st.stop()


    # --- Occupancy model loading ---
    occ_files = glob.glob("app/occupancy_xgb_pipeline_final.*")
    if not occ_files:
        occ_files = glob.glob("*.pkl") + glob.glob("*.joblib")  # fallback search
    
    if occ_files:
        file_path = occ_files[0]
        try:
            if file_path.endswith(".joblib"):
                models['occupancy'] = joblib.load(file_path)
                loading_status['occupancy'] = f"✅ Loaded ({file_path})"
            elif file_path.endswith(".pkl"):
                with open(file_path, 'rb') as f:
                    models['occupancy'] = pickle.load(f)
                loading_status['occupancy'] = f"✅ Loaded ({file_path})"
        except Exception as e:
            st.error(f"❌ Failed to load occupancy model: {str(e)}")
            st.stop()
    else:
        st.error("❌ No occupancy model file found (.pkl or .joblib)")
        st.stop()

    return models, loading_status

def rebuild_model_pipeline(model_name):
    """Attempt to rebuild a model pipeline with universal transformers"""
    # This is a fallback that creates a simple pipeline
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    
    # Create a simple pipeline with universal transformers
    pipeline = Pipeline([
        ('preprocessor', UniversalBinaryMapper()),
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    return pipeline

models, model_status = load_models()

# ====================== DATA LOADING (ORIGINAL APPROACH) ======================
@st.cache_data
def load_data():
    data = {
        'neighborhood': pd.DataFrame(),
        'listings': pd.DataFrame(),
        'kmeans': None,
        'scaler': None,
        'map_html': None
    }
    
    file_status = {}
    
    try:
        data['neighborhood'] = pd.read_csv('app/neighborhood_data_with_clusters.csv')
        file_status['neighborhood'] = 'âœ… Loaded'
    except Exception as e:
        file_status['neighborhood'] = f'âŒ Error: {str(e)[:30]}'
    
    try:
        data['listings'] = pd.read_csv('app/merged_df.csv')
        if 'room_type_x' in data['listings'].columns:
            data['listings'] = data['listings'].rename(columns={'room_type_x': 'room_type'})
        file_status['listings'] = 'âœ… Loaded'
    except Exception as e:
        file_status['listings'] = f'âŒ Error: {str(e)[:30]}'
    
    try:
        data['kmeans'] = joblib.load('app/kmeans_model.joblib')
        data['scaler'] = joblib.load('app/kmeans_scaler.joblib')
        file_status['clustering'] = 'âœ… Loaded'
    except:
        try:
            data['kmeans'] = pickle.load(open('app/kmeans_model.pkl', 'rb'))
            data['scaler'] = pickle.load(open('app/kmeans_scaler.pkl', 'rb'))
            file_status['clustering'] = 'âœ… Loaded (pickle)'
        except Exception as e:
            file_status['clustering'] = f'âŒ Error: {str(e)[:30]}'
    
    try:
        with open('app/cluster_map.html', 'r', encoding='utf-8') as f:
            data['map_html'] = f.read()
        file_status['map'] = 'âœ… Loaded'
    except Exception as e:
        file_status['map'] = f'âŒ Error: {str(e)[:30]}'
    
    return data, file_status

data, file_status = load_data()

# ====================== HELPER FUNCTIONS ======================
def get_training_feature_columns(model_type):
    """Get the actual feature columns from training data"""
    if model_type == 'price' and not feature_data['price_features'].empty:
        return feature_data['price_features'].columns.tolist()
    elif model_type == 'occupancy' and not feature_data['occupancy_features'].empty:
        return feature_data['occupancy_features'].columns.tolist()
    return []

def create_model_input_from_training_structure(user_inputs, model_type):
    if model_type == 'price':
        numeric = numeric_features_price
        binary = binary_features_price
        categorical = categorical_features_price
    elif model_type == 'occupancy':
        numeric = numeric_features_occupancy
        binary = binary_features_occupancy
        categorical = categorical_features_occupancy
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    features = numeric + binary + categorical

    row = {}
    for col in features:
        if col in numeric:
            row[col] = float(user_inputs.get(col, 0))
        elif col in binary:
            val = user_inputs.get(col, 0)
            row[col] = 't' if val in [1, 't', True] else 'f'
        elif col in categorical:
            row[col] = user_inputs.get(col, 'Unknown')
    return pd.DataFrame([row], columns=features)


def safe_predict(
    model,
    input_df: pd.DataFrame,
    model_name: str = "Model",
    inverse_log: bool = False,              # True if target was log(price+1)
    expected_columns: Optional[List] = None
):
    """
    Safely make a single prediction from a sklearn pipeline.
    
    Parameters
    ----------
    model : trained sklearn pipeline
        Your fitted pipeline (ColumnTransformer + estimator).
    input_df : pd.DataFrame
        One-row DataFrame with raw features matching training format.
    model_name : str
        For status messages, e.g., "Price" or "Occupancy".
    inverse_log : bool
        Set True if the target was log-transformed with np.log(price+1).
    expected_columns : list
        If provided, will reorder input_df and raise an error if any are missing.
    """
    try:
        # --- Validate shape and type ---
        if not isinstance(input_df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(input_df)}")

        if input_df.shape[0] != 1:
            raise ValueError(f"Expected 1 row, got {input_df.shape[0]}")

        # --- Optional: reorder columns to match training ---
        if expected_columns is not None:
            missing = [c for c in expected_columns if c not in input_df.columns]
            if missing:
                raise ValueError(f"Missing columns for {model_name}: {missing}")
            input_df = input_df[expected_columns]  # exact order

        # --- Predict ---
        pred = model.predict(input_df)

        # --- Extract scalar ---
        y_arr = np.asarray(pred).reshape(-1)
        if y_arr.size == 0:
            raise ValueError("Empty prediction output")
        yhat = float(y_arr[0])

        # --- Inverse log if needed ---
        if inverse_log:
            # match your training: log(price+1) => exp(...) - 1
            yhat = float(np.exp(yhat) - 1)

        return yhat, f"✅ {model_name} prediction successful"

    except Exception as e:
        return None, f"❌ {model_name} prediction failed: {str(e)[:200]}"

def get_neighborhood_stats(neighborhood_name):
    """Get statistics for a specific neighborhood"""
    if data['listings'].empty:
        return None
    
    neighborhood_data = data['listings'][
        data['listings']['neighbourhood_cleansed'] == neighborhood_name
    ]
    
    if neighborhood_data.empty:
        return None
    
    stats = {
        'avg_price': neighborhood_data['price'].mean() if 'price' in neighborhood_data.columns else 0,
        'total_listings': len(neighborhood_data),
        'avg_rating': neighborhood_data['review_scores_rating'].mean() if 'review_scores_rating' in neighborhood_data.columns else 4.5,
        'property_types': neighborhood_data['property_type'].value_counts().to_dict()
    }
    
    return stats

# ====================== UI COMPONENTS ======================
st.markdown('<div class="main-header"><h1>🏖️ Airbnb Cape Town Analysis & Prediction</h1></div>', unsafe_allow_html=True)

# Create tabs for better organization
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predictions", "🗺️ Map Analysis", "📊 Market Insights", "⚙️ System Status"])

with tab1:
    # Sidebar - Advanced Inputs
    with st.sidebar:
        st.header("🏠 Property Details")
        
        # Basic properties
        property_type = st.selectbox("Property Type", options=DEFAULT_VALUES['property_types'])
        room_type = st.selectbox("Room Type", options=DEFAULT_VALUES['room_types'])
        
        col1, col2 = st.columns(2)
        with col1:
            accommodates = st.slider("Accommodates", 1, 16, 4)
            bedrooms = st.slider("Bedrooms", 0, 8, 2)
        with col2:
            bathrooms = st.slider("Bathrooms", 1, 5, 2)
            beds = st.slider("Beds", 1, 10, max(bedrooms, 1))
        
        neighbourhood = st.selectbox("Neighbourhood", options=DEFAULT_VALUES['neighbourhoods'])
        
        st.subheader("🌟 Host & Property Features")
        host_is_superhost = st.checkbox("Host is Superhost")
        
        col3, col4 = st.columns(2)
        with col3:
            number_of_reviews = st.number_input("Number of Reviews", 0, 1000, 25)
            host_response_rate = st.slider("Host Response Rate", 0.0, 1.0, 0.95, 0.05)
        with col4:
            avg_rating = st.slider("Average Rating", 1.0, 5.0, 4.5, 0.1)
            calculated_host_listings_count = st.number_input("Host Total Listings", 1, 100, 1)
        
        st.subheader("📅 Booking Details")
        col5, col6 = st.columns(2)
        with col5:
            minimum_nights = st.slider("Minimum Nights", 1, 30, 2)
            availability_rate = st.slider("Availability Rate", 0.0, 1.0, 0.7, 0.05)
        with col6:
            days_since_last_review = st.slider("Days Since Last Review", 0, 365, 30)

    # Main prediction area
    st.header("🔮 Property Performance Prediction")
    
    if st.button("🚀 Generate Predictions", type="primary", use_container_width=True):
        # Collect all user inputs
        user_inputs = {
            'property_type': property_type,
            'room_type': room_type,
            'accommodates': accommodates,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'neighbourhood_cleansed': neighbourhood,
            'host_is_superhost': host_is_superhost,
            'number_of_reviews': number_of_reviews,
            'host_response_rate': host_response_rate,
            'beds': beds,
            'avg_rating': avg_rating,
            'calculated_host_listings_count': calculated_host_listings_count,
            'minimum_nights': minimum_nights,
            'availability_rate': availability_rate,
            'days_since_last_review': days_since_last_review
        }
        
                # Show feature structure info
        price_features = get_training_feature_columns('price')
        occupancy_features = get_training_feature_columns('occupancy')

        if price_features:
            st.info(f"Using price model with {len(price_features)} features from training data")
        if occupancy_features:
            st.info(f"Using occupancy model with {len(occupancy_features)} features from training data")

        # Initialize predictions with fallback values
        price_pred = 1200
        occupancy_pred = 150
        confidence_price = "Model not available"
        confidence_occupancy = "Model not available"

        # ======== Validate models and apply fallback if needed ========
        from sklearn.base import BaseEstimator

        def is_valid_model(model):
            return isinstance(model, BaseEstimator)

        if not is_valid_model(models['price']):
            st.warning("Price model is invalid (type: {}). Using fallback pipeline.".format(type(models['price'])))
            models['price'] = rebuild_model_pipeline("price")

        if not is_valid_model(models['occupancy']):
            st.warning("Occupancy model is invalid (type: {}). Using fallback pipeline.".format(type(models['occupancy'])))
            models['occupancy'] = rebuild_model_pipeline("occupancy")

        # ======== Price prediction using saved model with training data structure ========
        if models['price'] is not None:
            price_input = create_model_input_from_training_structure(user_inputs, 'price')

            if price_input is not None:
                price_pred_val, status = safe_predict(
                    models['price'],
                    price_input,
                    model_name="Price",
                    inverse_log=True,  # ✅ you trained on log(price + 1)
                    expected_columns=(numeric_features_price + binary_features_price + categorical_features_price)
                )
            
                if price_pred_val is not None:
                    price_pred = max(float(price_pred_val), 50.0)
                    confidence_price = "High confidence (using saved model)"
                else:
                    price_pred = None
                    confidence_price = status
                    st.warning(f"Price prediction issue: {status}")

        # ======== Occupancy prediction using saved model with training data structure ========
        if models['occupancy'] is not None:
            user_inputs_with_price = user_inputs.copy()
            
            if 'avg_price' in numeric_features_occupancy:
                if 'price_pred' in locals() and price_pred is not None:
                    user_inputs_with_price['avg_price'] = float(price_pred)
                else:
                    user_inputs_with_price['avg_price'] = float(user_inputs.get('avg_price', 0))

            occupancy_input = create_model_input_from_training_structure(user_inputs_with_price, 'occupancy')
            if occupancy_input is not None:
                occ_pred_val, status = safe_predict(
                    models['occupancy'],
                    occupancy_input,
                    model_name="Occupancy",
                    inverse_log=False,
                    expected_columns=(numeric_features_occupancy + binary_features_occupancy + categorical_features_occupancy)
                    )
                if occ_pred_val is not None:
                    occupancy_pred = float(min(max(occ_pred_val, 0.0), 365.0))
                    confidence_occupancy = "High confidence (using saved model)"
                else:
                    confidence_occupancy = status
                    st.warning(f"Occupancy prediction issue: {status}")

           
        
        # Calculate financial metrics
        annual_revenue = price_pred * occupancy_pred
        monthly_revenue = annual_revenue / 12
        occupancy_rate = (occupancy_pred / 365) * 100
        
        # Display results in an attractive format
        st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
        st.subheader("🎯 Prediction Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Daily Price",
                value=f"R{price_pred:,.0f}",
                help=f"Confidence: {confidence_price}"
            )
        
        with col2:
            st.metric(
                label="Annual Occupancy",
                value=f"{occupancy_pred:.0f} days",
                delta=f"{occupancy_rate:.1f}% rate",
                help=f"Confidence: {confidence_occupancy}"
            )
        
        with col3:
            st.metric(
                label="Annual Revenue",
                value=f"R{annual_revenue:,.0f}",
                help="Daily price × Annual occupancy"
            )
        
        with col4:
            st.metric(
                label="Monthly Revenue",
                value=f"R{monthly_revenue:,.0f}",
                help="Annual revenue ÷ 12"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Additional insights
        st.subheader("📈 Market Insights")
        neighborhood_stats = get_neighborhood_stats(neighbourhood)
        
        if neighborhood_stats:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Neighborhood Statistics for {neighbourhood}:**")
                st.write(f"• Average price in area: R{neighborhood_stats['avg_price']:.0f}")
                st.write(f"• Total listings: {neighborhood_stats['total_listings']}")
                st.write(f"• Average rating: {neighborhood_stats['avg_rating']:.1f}/5")
                
                # Price comparison
                if neighborhood_stats['avg_price'] > 0:
                    price_diff = ((price_pred - neighborhood_stats['avg_price']) / neighborhood_stats['avg_price']) * 100
                    if price_diff > 0:
                        st.success(f"Your predicted price is {price_diff:.1f}% above neighborhood average")
                    else:
                        st.info(f"Your predicted price is {abs(price_diff):.1f}% below neighborhood average")
            
            with col2:
                if neighborhood_stats['property_types'] is not None:
                    st.write("**Property Type Distribution:**")
                    property_df = pd.DataFrame(
                        list(neighborhood_stats['property_types'].items()),
                        columns=['Property Type', 'Count']
                    )
                    fig = px.pie(
                        property_df, 
                        values='Count', 
                        names='Property Type',
                        title=f"Property Types in {neighbourhood}"
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("🗺️ Neighborhood Cluster Analysis")
    
    if data['map_html'] is not None:
        st.components.v1.html(data['map_html'], height=600)
    elif not data['neighborhood'].empty:
        # Create fallback map
        center_lat = data['neighborhood']['latitude'].mean() if 'latitude' in data['neighborhood'].columns else -33.9249
        center_lon = data['neighborhood']['longitude'].mean() if 'longitude' in data['neighborhood'].columns else 18.4241
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
        
        # Add markers for neighborhoods
        for _, row in data['neighborhood'].iterrows():
            lat = row.get('latitude', center_lat)
            lon = row.get('longitude', center_lon)
            name = row.get('neighbourhood', 'Unknown')
            cluster = row.get('cluster', 0)
            
            # Color based on cluster
            colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']
            color = colors[cluster % len(colors)]
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                popup=f"<b>{name}</b><br>Cluster: {cluster}",
                tooltip=name
            ).add_to(m)
        
        # Add legend
        legend_html = """
        <div style="position: fixed; top: 10px; right: 10px; background-color: white; border: 2px solid grey; z-index:9999; padding: 10px">
        <p><b>Neighborhood Clusters</b></p>
        """
        for i, color in enumerate(colors[:5]):  # Show first 5 clusters
            legend_html += f'<p><i class="fa fa-circle" style="color:{color}"></i> Cluster {i}</p>'
        legend_html += "</div>"
        
        m.get_root().html.add_child(folium.Element(legend_html))
        
        folium_static(m, width=1200, height=600)
    else:
        st.warning("No map data available. Please check if the required data files are loaded.")

with tab3:
    st.header("📊 Market Insights Dashboard")
    
    if not data['listings'].empty:
        # Market overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_price = data['listings']['price'].mean() if 'price' in data['listings'].columns else 0
            st.metric("Average Price", f"R{avg_price:,.0f}")
        
        with col2:
            total_listings = len(data['listings'])
            st.metric("Total Listings", f"{total_listings:,}")
        
        with col3:
            avg_rating = data['listings']['avg_rating'].mean() if 'avg_rating' in data['listings'].columns else 0
            st.metric("Average Rating", f"{avg_rating:.1f}/5")
        
        with col4:
            # Fixed the type error here
            superhosts = 0
            superhost_rate = 0
            
            if 'host_is_superhost' in data['listings'].columns:
                # Ensure we're working with numeric values
                superhost_column = data['listings']['host_is_superhost']
                
                # Handle different data types
                if superhost_column.dtype == 'bool':
                    superhosts = superhost_column.sum()
                elif superhost_column.dtype == 'object':
                    # Handle string values like 't', 'f', 'True', 'False'
                    superhosts = sum(1 for x in superhost_column if str(x).lower() in ['t', 'true', '1', 'yes'])
                else:
                    # Numeric values
                    try:
                        superhosts = int(pd.to_numeric(superhost_column, errors='coerce').sum())
                    except:
                        superhosts = 0
                
                superhost_rate = (superhosts / total_listings) * 100 if total_listings > 0 else 0
            
            st.metric("Superhost Rate", f"{superhost_rate:.1f}%")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            if 'property_type' in data['listings'].columns:
                property_counts = data['listings']['property_type'].value_counts().head(10)
                fig = px.bar(
                    x=property_counts.values,
                    y=property_counts.index,
                    orientation='h',
                    title="Top Property Types",
                    labels={'x': 'Count', 'y': 'Property Type'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'room_type' in data['listings'].columns:
                room_counts = data['listings']['room_type'].value_counts()
                fig = px.pie(
                    values=room_counts.values,
                    names=room_counts.index,
                    title="Room Type Distribution"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Price analysis
        if 'price' in data['listings'].columns and 'neighbourhood_cleansed' in data['listings'].columns:
            st.subheader("💰 Price Analysis by Neighborhood")
            
            neighborhood_prices = data['listings'].groupby('neighbourhood_cleansed')['price'].agg(['mean', 'count']).reset_index()
            neighborhood_prices = neighborhood_prices[neighborhood_prices['count'] >= 5]  # Filter neighborhoods with at least 5 listings
            neighborhood_prices = neighborhood_prices.sort_values('mean', ascending=False).head(15)
            
            fig = px.bar(
                neighborhood_prices,
                x='neighbourhood_cleansed',
                y='mean',
                title="Average Price by Neighborhood (Top 15)",
                labels={'mean': 'Average Price (R)', 'neighbourhood_cleansed': 'Neighborhood'}
            )
            fig.update_xaxes(tickangle=45)
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("No listing data available for market insights.")

with tab4:
    st.header("⚙️ System Status & Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Saved Model Status")
        for name, status in model_status.items():
            if "✅" in status:
                st.success(f"{name.title()}: {status}")
            else:
                st.error(f"{name.title()}: {status}")
    
    with col2:
        st.subheader("📁 Training Feature Files")
        for name, status in feature_file_status.items():
            if "✅" in status:
                st.success(f"{name.replace('_', ' ').title()}: {status}")
            else:
                st.error(f"{name.replace('_', ' ').title()}: {status}")
    
    # Show feature structure comparison
    st.subheader("🔧 Feature Structure")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Price Model Features:**")
        price_features = get_training_feature_columns('price')
        if price_features:
            st.write(f"Total features: {len(price_features)}")
            with st.expander("View all price features"):
                for i, feature in enumerate(price_features, 1):
                    st.write(f"{i}. {feature}")
        else:
            st.error("No price training features loaded")
    
    with col2:
        st.write("**Occupancy Model Features:**")
        occupancy_features = get_training_feature_columns('occupancy')
        if occupancy_features:
            st.write(f"Total features: {len(occupancy_features)}")
            with st.expander("View all occupancy features"):
                for i, feature in enumerate(occupancy_features, 1):
                    st.write(f"{i}. {feature}")
        else:
            st.error("No occupancy training features loaded")
    
    # Additional data status
    st.subheader("📊 Additional Data Files")
    for name, status in file_status.items():
        if "✅" in status:
            st.success(f"{name.title()}: {status}")
        else:
            st.error(f"{name.title()}: {status}")
    
    # Show custom transformer status
    st.write(f"**Custom Transformers:** {custom_transformer_status}")
    
    # Troubleshooting section
    st.subheader("🔧 File Requirements")
    with st.expander("Required Files & Format"):
        st.write("""
        **Saved Models (Original):**
        - `price_prediction_model.joblib` or `price_prediction_model.pkl`
        - `occupancy_xgb_pipeline.joblib` or `occupancy_xgb_pipeline.pkl`
        
        **Training Feature Files (New):**
        - `train_price_data.csv` - Features + 'price' target column
        - `train_occupancy_data.csv` - Features + 'occupancy' target column
        
        **Additional Files (Optional):**
        - `merged_df.csv` - For market insights
        - `neighborhood_data_with_clusters.csv` - For map analysis
        - `cluster_map.html` - For interactive map
        
        **How it works:**
        1. App loads your saved/trained models (joblib/pkl files)
        2. App loads training feature CSVs to understand exact feature structure
        3. User inputs are mapped to match training feature structure
        4. Predictions are made using saved models with properly structured input
        
        **Benefits:**
        - Uses your actual trained models (no retraining)
        - Ensures feature consistency with training data
        - Handles log-transformed prices automatically
        - Maintains model performance
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "🏖️ Airbnb Cape Town Analysis Dashboard | Built with Streamlit & ❤️ | Using Saved Models + Training Features"
    "</div>",
    unsafe_allow_html=True
)
