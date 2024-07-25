from text_model import TextModel
from image_model import ImageModel
import numpy as np

class CombinedModel:
    def __init__(self):
        self.text_model = TextModel()
        self.image_model = ImageModel()

    def predict(self, text, image):
        text_pred = self.text_model.predict(text)
        image_pred = self.image_model.predict(image)
        combined_pred = np.mean([text_pred, image_pred], axis=0)
        return combined_pred
