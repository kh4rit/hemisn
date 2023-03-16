import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

if not torch.backends.mps.is_available():
    print ("MPS device not found.")

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

    def forward(self, input_ids, attention_mask=None):
        # The variable input_tokens is defined in the forward method of the HybridMemoryModel, but it is not used anywhere
        # Tokenize the input text
        # input_tokens = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        
        # Obtain contextualized embeddings from the transformer
        transformer_output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
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
    input_tokens = input_tokens = model.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
    start_position = torch.argmax(start_logits, dim=-1)
    end_position = torch.argmax(end_logits, dim=-1)
    
    answers = []
    for i, (start, end) in enumerate(zip(start_position, end_position)):
        answer_tokens = input_tokens['input_ids'][i][start:end + 1]
        answer = model.tokenizer.decode(answer_tokens)
        answers.append(answer)
    
    return answers

pretrained_model_name = "bert-base-uncased"
memory_size = 128

model = HybridMemoryModel(pretrained_model_name, memory_size)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

input_text = ["What is the capital of France?", "Who wrote 'The Catcher in the Rye'?"]
input_ids = model.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)["input_ids"]

input_ids = input_ids.to(device)

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

from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def prepare_data(context, question, answer):
    max_length = 512  # BERT model's max length

    tokenized_context = tokenizer.encode_plus(
        context,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    tokenized_question = tokenizer.encode_plus(
        question,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )

    # Combine tokenized_context and tokenized_question
    input_ids = (
        [tokenizer.cls_token_id]
        + tokenized_question["input_ids"]
        + [tokenizer.sep_token_id]
        + tokenized_context["input_ids"]
        + [tokenizer.sep_token_id]
    )
    offsets = (
        [(0, 0)]
        + tokenized_question["offset_mapping"]
        + [(0, 0)]
        + tokenized_context["offset_mapping"]
        + [(0, 0)]
    )

    # Truncate or pad the input_ids and offsets to max_length
    input_ids = input_ids[:max_length] + [tokenizer.pad_token_id] * (max_length - len(input_ids))
    offsets = offsets[:max_length] + [(0, 0)] * (max_length - len(offsets))

    answer_start = context.find(answer)
    answer_end = answer_start + len(answer)

    if answer_start == -1:
        return None, None, None

    token_start_position, token_end_position = None, None
    for i, (offset_start, offset_end) in enumerate(offsets):
        if offset_start <= answer_start <= offset_end:
            token_start_position = i
        if offset_start <= answer_end <= offset_end:
            token_end_position = i
            break

    if token_start_position is None or token_end_position is None:
        return None, None, None

    return input_ids, token_start_position, token_end_position

def preprocess_data(example):
    context = example["context"]
    question = example["question"]
    answer = example["answers"]["text"][0]

    input_text, token_start_position, token_end_position = prepare_data(context, question, answer)

    if input_text is None or token_start_position is None or token_end_position is None:
        return {
            "input_ids": None,
            "attention_mask": None,
            "start_position": None,
            "end_position": None,
        }

    input_ids = torch.tensor(input_text, dtype=torch.long).unsqueeze(0)
    attention_mask = (input_ids != 0).float()
    encoding = {"input_ids": input_ids, "attention_mask": attention_mask}
    input_ids = encoding["input_ids"].squeeze()
    attention_mask = encoding["attention_mask"].squeeze()

    return {
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "start_position": token_start_position,
        "end_position": token_end_position,
    }


# Preprocess the dataset
train_percentage = 0.05     # We use only 1%, only testing for now
train_size = int(len(dataset["train"]) * train_percentage)
train_data = dataset["train"].select(range(train_size)).map(preprocess_data, remove_columns=['id', 'title', 'context', 'question', 'answers'], load_from_cache_file=False).filter(lambda example: example['input_ids'] is not None)
# Prepare a DataLoader for the training data
# ==========================================

from torch.utils.data import DataLoader

from torch.nn.utils.rnn import pad_sequence

train_dataset = torch.utils.data.TensorDataset(
    pad_sequence([torch.tensor(ids) for ids in train_data["input_ids"]], batch_first=True),
    pad_sequence([torch.tensor(mask) for mask in train_data["attention_mask"]], batch_first=True),
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

from tqdm import tqdm

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")

    progress_bar = tqdm(train_dataloader, desc="Training", unit="batch")
    
    model.train()
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids, attention_mask, start_position, end_position = [x.to(device) for x in batch]
        start_logits, end_logits = model(input_ids=input_ids, attention_mask=attention_mask)
        
        loss_start = nn.CrossEntropyLoss()(start_logits, start_position)
        loss_end = nn.CrossEntropyLoss()(end_logits, end_position)
        total_loss = (loss_start + loss_end) / 2

        total_loss.backward()
        optimizer.step()
        
        # Update the progress bar description with the current loss, accuracy, or any other metric
        progress_bar.set_description(f"Training (loss={total_loss:.4f})")

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss.item()}")


# Test again the input text

input_text = ["What is the capital of France?", "Who wrote 'The Catcher in the Rye'?"]
input_ids = model.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)["input_ids"]

input_ids = input_ids.to(device)

start_logits, end_logits = model(input_ids=input_ids)

answers = decode_answer(input_text, start_logits, end_logits)
print(answers)
