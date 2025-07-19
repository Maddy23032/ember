import lief
import ember
import numpy as np

def extract_vector(file_path, feature_version=2):
    lief.logging.set_level(lief.logging.LOGGING_LEVEL.ERROR)
    raw_features = ember.extract_raw_features(file_path, feature_version)
    extractor = ember.PEFeatureExtractor(feature_version=feature_version)
    vector = extractor.process_raw_features(raw_features)
    return np.array(vector, dtype=np.float32).reshape(1, -1)
