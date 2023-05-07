import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
import random
import tqdm
import os
# 使用对比学习训练一个BERT模型。通过学习将相似的文本投影到相似的表示空间，模型可以学会捕捉文本之间的关系。
# 设置transformers库中环境变量，强制使用utf-8编码
os.environ["PYTHONIOENCODING"] = "utf-8"

class ContrastiveDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    # 返回数据集中文本数量
    def __len__(self):
        return len(self.texts)
    # 使用分词器将文本转化为模型输入格式
    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze()

class ContrastiveModel(nn.Module):
    def __init__(self, base_model_name, projection_dim=128):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name, from_tf=False)
        self.projection = nn.Linear(self.encoder.config.hidden_size, projection_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state[:, 0]
        projections = self.projection(hidden_states)
        return projections



def contrastive_loss(projections, temperature=0.5):
    device = projections.device
    batch_size = projections.size(0)
    similarity_matrix = torch.matmul(projections, projections.T) / temperature
    labels = torch.arange(batch_size).to(device)
    return nn.CrossEntropyLoss()(similarity_matrix, labels)

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for input_ids, attention_mask in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        projections = model(input_ids, attention_mask)
        loss = contrastive_loss(projections)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def load_texts_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]
    return texts

file_path = 'your_text_file.txt'
texts = load_texts_from_file(file_path)
dataset = ContrastiveDataset(texts, tokenizer)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Replace with your own text dataset
    texts = [
        "Natural language processing is a subfield of linguistics.",
        "The history of NLP generally starts in the 1950s.",
        "Machine learning algorithms for language processing.",
        "Deep learning-based approaches to NLP."
    ]

    dataset = ContrastiveDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = ContrastiveModel(base_model_name).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    num_epochs = 100
    for epoch in range(num_epochs):
        epoch_loss = train_epoch(model, dataloader, optimizer, device)
        print(f"Epoch: {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

if __name__ == "__main__":
    main()
