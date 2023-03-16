import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class HybridMemoryModel(nn.Module):
    def __init__(self, pretrained_model_name, memory_size):
        super(HybridMemoryModel, self).__init__()
        self.transformer = AutoModel.from_pretrained(pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        
        hidden_size = self.transformer.config.hidden_size
        self.memory = nn.Parameter(torch.randn(memory_size, hidden_size))
        self.memory_attention = nn.Linear(hidden_size, memory_size)
        
        self.start_predictor = nn.Linear(hidden_size, 1)
        self.end_predictor = nn.Linear(hidden_size, 1)

    def forward(self, input_ids):
        # Tokenize the input text
        input_tokens = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        
        # Obtain contextualized embeddings from the transformer
        transformer_output = self.transformer(input_ids=input_ids)
        hidden_states = transformer_output.last_hidden_state
        
        # Calculate attention scores between hidden states and memory
        attention_scores = self.memory_attention(hidden_states)
        
        # Apply softmax to get attention probabilities
        attention_probs = torch.softmax(attention_scores, dim=-1)
        
        # Retrieve memory-aware hidden states
        memory_aware_hidden_states = attention_probs.matmul(self.memory)
        
        # Combine the original hidden states and memory-aware hidden states
        combined_hidden_states = hidden_states + memory_aware_hidden_states
        
        # Predict start and end positions
        start_logits = self.start_predictor(combined_hidden_states).squeeze(-1)
        end_logits = self.end_predictor(combined_hidden_states).squeeze(-1)
        
        return start_logits, end_logits

def decode_answer(input_text, start_logits, end_logits):
    input_tokens = model.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    start_positions = torch.argmax(start_logits, dim=-1)
    end_positions = torch.argmax(end_logits, dim=-1)
    
    answers = []
    for i, (start, end) in enumerate(zip(start_positions, end_positions)):
        answer_tokens = input_tokens['input_ids'][i][start:end + 1]
        answer = model.tokenizer.decode(answer_tokens)
        answers.append(answer)
    
    return answers

pretrained_model_name = "bert-base-uncased"
memory_size = 128

model = HybridMemoryModel(pretrained_model_name, memory_size)
input_text = ["What is the capital of France?", "Who wrote 'The Catcher in the Rye'?"]
input_ids = model.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)["input_ids"]
start_logits, end_logits = model(input_ids=input_ids)

answers = decode_answer(input_text, start_logits, end_logits)
print(answers)


# load the SQuAD dataset and preprocess it
# ========================================

from datasets import load_dataset

# Load the SQuAD dataset
dataset = load_dataset("squad")

def char_to_token_indices(context, answer_start, answer_end, tokenizer, max_length):
    context_tokens = tokenizer.tokenize(context)
    char_to_token = []
    token_start = 0
    for token in context_tokens:
        token_end = token_start + len(token)
        char_to_token.extend([token_start] * len(token))
        token_start = token_end

    # Check if the character indices are within the bounds of the char_to_token list
    if (
        answer_start >= len(char_to_token)
        or answer_end >= len(char_to_token)
        or char_to_token[answer_start] >= max_length
        or char_to_token[answer_end] >= max_length
    ):
        return None, None

    token_start_index = char_to_token[answer_start]
    token_end_index = char_to_token[answer_end]

    return token_start_index, token_end_index

def preprocess_data(example):
    input_text = example["context"] + " " + example["question"]
    start_position = example["answers"]["answer_start"][0]
    end_position = start_position + len(example["answers"]["text"][0])
    
    # Convert character indices to token indices
    token_start_position, token_end_position = char_to_token_indices(
        example["context"], start_position, end_position, model.tokenizer, max_length=512
    )

    if token_start_position is None or token_end_position is None:
        return {
            "input_ids": None,
            "start_position": None,
            "end_position": None
        }

    # Get input_ids
    input_ids = model.tokenizer.encode(input_text, max_length=512, padding="max_length", truncation=True)

    return {
        "input_ids": input_ids,
        "start_position": token_start_position,
        "end_position": token_end_position
    }

# Preprocess the dataset
train_data = dataset["train"].map(preprocess_data, remove_columns=['id', 'title', 'context', 'question', 'answers'], load_from_cache_file=False)
train_data = train_data.filter(lambda x: x['input_ids'] is not None)

# Prepare a DataLoader for the training data
# ==========================================

from torch.utils.data import DataLoader

train_dataset = torch.utils.data.TensorDataset(
    torch.stack(train_data["input_ids"]),
    torch.tensor(train_data["start_position"]),
    torch.tensor(train_data["end_position"])
)

batch_size = 8
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Set up the training loop and fine-tune the model
# ================================================

from torch.optim import Adam

# Set up the optimizer
optimizer = Adam(model.parameters(), lr=3e-5)

# Training loop
epochs = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()
        
        input_ids, attention_mask, start_positions, end_positions = [x.to(device) for x in batch]
        start_logits, end_logits = model(input_ids=input_ids)
        
        loss_start = nn.CrossEntropyLoss()(start_logits, start_positions)
        loss_end = nn.CrossEntropyLoss()(end_logits, end_positions)
        total_loss = (loss_start + loss_end) / 2

        total_loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss.item()}")


# Test again the input text

input_text = ["What is the capital of France?", "Who wrote 'The Catcher in the Rye'?"]
input_ids = model.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)["input_ids"]
start_logits, end_logits = model(input_ids=input_ids)

answers = decode_answer(input_text, start_logits, end_logits)
print(answers)
