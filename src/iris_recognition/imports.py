# Standard library imports
import os
import random
import json

# Third-party imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Custom modules imports
from iris_recognition.preprocessing.iris_localization import IrisLocalizer
from iris_recognition.preprocessing.iris_normalization import IrisNormalizer
from iris_recognition.preprocessing.iris_illumination import IrisIlluminater
from iris_recognition.preprocessing.iris_enhancement import IrisEnhancer
from iris_recognition.feature_extraction.feature_extraction import FeatureExtractor
from iris_recognition.matching.iris_matching import IrisMatcher
from iris_recognition.model.iris_preprocessing import IrisDataPreprocessor
from iris_recognition.model.performance_evaluation import PerformanceEvaluator
from iris_recognition.utils.data_loading import DataLoader