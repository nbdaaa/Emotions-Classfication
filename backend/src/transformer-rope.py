import torch
import tiktoken
from Model.TransformerWithROPE.TransformerEncoderROPE import TransformerModelWithROPE

best_model_path = 'Saved trained model/best_transformerROPE_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
tokenizer = tiktoken.get_encoding('gpt2')
# Initialize the Transformer model
vocab_size = tokenizer.n_vocab
embed_size = 256
d_model = 256
num_heads = 8
d_ff = 512
output_size = 6
num_layers = 3
dropout = 0.2

# Initialize model and move to device
#model = TransformerModel(vocab_size, embed_size, d_model, num_heads, d_ff, output_size, num_layers, dropout)
model = TransformerModelWithROPE(vocab_size, embed_size, d_model, num_heads, d_ff, output_size, num_layers, dropout)
model = model.to(device)

# Assuming the same model architecture
model.load_state_dict(torch.load(best_model_path))
model.eval()  # Set to evaluation mode if using for inference

label_dict = {4: 'sadness', 2: 'joy', 0: 'anger', 3: 'love', 5: 'surprise', 1: 'fear'}

def predict_emotion(sentence, model, tokenizer, device):
    # Preprocess the input sentence
    sentence = sentence.lower()
    encoding = tokenizer.encode(sentence)
    input_ids = torch.tensor(encoding, dtype=torch.long, device=device).unsqueeze(0)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
        accuracy, predicted = torch.max(outputs, 1)
          
    return label_dict[predicted.item()]

if __name__ == "__main__":
    # Example usage
    test_sentence = input("Enter a sentence to predict its emotion: ")
    predicted_emotion = predict_emotion(test_sentence, model, tokenizer, device)
    print(f"Predicted emotion: {predicted_emotion}")