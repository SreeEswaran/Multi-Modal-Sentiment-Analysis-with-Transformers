from transformers import BertTokenizer, BertForSequenceClassification
import torch

class TextModel:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        return torch.softmax(outputs.logits, dim=1).numpy()
