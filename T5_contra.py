import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
from datasets import load_dataset
from tqdm import tqdm

# 加载预训练的 T5-small 模型和分词器
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 设置 T5-small 的强制开始标记
model.config.forced_bos_token_id = tokenizer.eos_token_id

# 定义对比学习损失函数（InfoNCE 损失）
def contrastive_loss(anchor, positive, negatives, temperature=0.07):
    anchor = anchor.view(1, -1)
    logits = torch.cat((anchor @ positive.t(), anchor @ negatives.t()), dim=1) / temperature
    return nn.CrossEntropyLoss()(logits, torch.tensor([0], device=device))

class SquadDataset(Dataset):
    def __init__(self, squad_data, tokenizer, max_length):
        self.squad_data = squad_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.squad_data)

    def __getitem__(self, idx):
        data = self.squad_data[idx]
        context = data["context"]
        answer = data["answers"]["text"][0]

        # 将问题和答案拼接成 T5 输入格式
        input_text = f"context: {context} answer: {answer}"
        input_encoding = tokenizer(input_text, max_length=self.max_length, padding="max_length", truncation=True,
                                   return_tensors="pt")

        # 对问题进行编码
        question = data["question"]
        question_encoding = tokenizer(question, max_length=self.max_length, padding="max_length", truncation=True,
                                      return_tensors="pt")

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "question_ids": question_encoding["input_ids"].squeeze(),
            "question_attention_mask": question_encoding["attention_mask"].squeeze()
        }


# 加载 SQuAD 数据集
squad_dataset = load_dataset("squad")["train"]

# 创建数据集
max_length = 512
train_dataset = SquadDataset(squad_data=squad_dataset, tokenizer=tokenizer, max_length=max_length)

# 创建数据加载器
batch_size = 8
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def train(model, dataloader, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        # 使用 tqdm 包装 dataloader
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch in progress_bar:
            # 从批次中提取数据并将其传输到设备上
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            question_ids = batch["question_ids"].to(device)
            question_attention_mask = batch["question_attention_mask"].to(device)

            # 生成问题
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=5)

            # 计算对比学习损失
            anchor = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
            positive = model(input_ids=question_ids, attention_mask=question_attention_mask).last_hidden_state[:, 0, :]
            negatives = model(input_ids=outputs, attention_mask=(outputs != tokenizer.pad_token_id).long()).last_hidden_state[:, 0, :]
            loss = contrastive_loss(anchor, positive, negatives)

            # 反向传播和优化
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 更新进度条
            progress_bar.set_postfix({"Loss": loss.item()})

# 设置优化器
optimizer = AdamW(model.parameters(), lr=5e-5)

# 训练模型
num_epochs = 10
train(model, dataloader, optimizer, num_epochs)