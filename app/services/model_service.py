import joblib
import pandas as pd
from app.core.config import settings
from app.cache.redis_cache import set_cache_predictoin , get_cache_prediction

model = joblib.load(settings.MODEL_PATH)

def predict_car_price(data : dict):
    cache_key = " ".join([str(val) for val in data.values])
    cached_prediction = get_cache_prediction(cache_key)
    if cached_prediction:
        return cached_prediction
    
    input_data = pd.DataFrame(data=data)
    prediction = model.predict(input_data)[0]
    set_cache_predictoin(cache_key,prediction)
    return prediction
