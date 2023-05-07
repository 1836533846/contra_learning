import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BartForConditionalGeneration, BartTokenizer, AdamW
from datasets import load_dataset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# 加载预训练的 BART 模型和分词器
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)

# 设置 BART 的强制开始标记
model.config.forced_bos_token_id = tokenizer.eos_token_id

# 定义对比学习损失函数（InfoNCE 损失）
def contrastive_loss(anchor, positive, negatives, temperature=0.07):
    anchor = anchor.view(1, -1)
    logits = torch.cat((anchor @ positive.t(), anchor @ negatives.t()), dim=1) / temperature
    return nn.CrossEntropyLoss()(logits, torch.tensor([0], device=device))

# 定义 SquadDataset 类
class SquadDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        context = sample["context"]
        question = sample["question"]
        answer = sample["answers"]["text"][0]

        input_encoding = self.tokenizer(f"context: {context} answer: {answer}", truncation=True, padding="max_length", max_length=self.max_length)
        question_encoding = self.tokenizer(question, truncation=True, padding="max_length", max_length=self.max_length)

        return {
            "input_ids": torch.tensor(input_encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(input_encoding["attention_mask"], dtype=torch.long),
            "question_ids": torch.tensor(question_encoding["input_ids"], dtype=torch.long),
            "question_attention_mask": torch.tensor(question_encoding["attention_mask"], dtype=torch.long),
        }

# 加载 SQuAD 数据集
squad_dataset = load_dataset("squad")["train"]

# 创建数据集
max_length = 512
train_dataset = SquadDataset(squad_dataset, tokenizer, max_length=max_length)

# 创建数据加载器
batch_size = 8
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 定义训练函数
def train(model, dataloader, optimizer, num_epochs):
    model.train()
    scaler = GradScaler()

    for epoch in range(num_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            question_ids = batch["question_ids"].to(device)
            question_attention_mask = batch["question_attention_mask"].to(device)

            with autocast():  # 使用混合精度（省显存）
                outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=5)

                anchor = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
                positive = model(input_ids=question_ids, attention_mask=question_attention_mask).last_hidden_state[:, 0, :]
                negatives = model(input_ids=outputs, attention_mask=(outputs != tokenizer.pad_token_id).long()).last_hidden_state[:, 0, :]
                loss = contrastive_loss(anchor, positive, negatives)

            optimizer.zero_grad()
            scaler.scale(loss).backward()  # 使用 GradScaler 计算梯度
            scaler.step(optimizer)  # 使用 GradScaler 更新权重
            scaler.update()  # 更新 GradScaler 的状态

            progress_bar.set_postfix({"Loss": loss.item()})



optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 10
train(model, dataloader, optimizer, num_epochs)

model.save_pretrained("trained_bart_base")

loaded_model = BartForConditionalGeneration.from_pretrained("trained_bart_base")
loaded_model.to(device)

def generate_question(context, answer, model, tokenizer, device):
    input_text = f"context: {context} answer: {answer}"
    input_encoding = tokenizer(input_text, return_tensors="pt")
    input_ids = input_encoding["input_ids"].to(device)
    attention_mask = input_encoding["attention_mask"].to(device)
    output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=5)
    question = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return question

context = "Neil Alden Armstrong was an American astronaut and aeronautical engineer who was the first person to walk on the Moon. He was also a naval aviator, test pilot, and university professor."
answer = "Neil Alden Armstrong"

question = generate_question(context, answer, loaded_model, tokenizer, device)
print("Generated Question:", question)
