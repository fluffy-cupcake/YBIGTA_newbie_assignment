import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from datasets import load_dataset

from tqdm import tqdm # type: ignore
from sklearn.metrics import f1_score

from word2vec import Word2Vec
from model import MyGRULanguageModel
from config import *


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size: int = tokenizer.vocab_size

    # load Word2Vec checkpoint
    word2vec = Word2Vec(vocab_size, d_model, window_size, method).to(device)
    checkpoint = torch.load("word2vec.pt", map_location=device)
    word2vec.load_state_dict(checkpoint)
    embeddings: Tensor = word2vec.embeddings_weight()

    # define model
    model = MyGRULanguageModel(d_model, hidden_size, num_classes, embeddings).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = load_dataset("google-research-datasets/poem_sentiment")
    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset["validation"], batch_size=batch_size)

    for epoch in tqdm(range(num_epochs)):
        model.train()
        loss_sum = 0.0
        for batch in train_loader:
            input_ids = tokenizer(batch["verse_text"], padding=True, return_tensors="pt")\
                .input_ids.to(device)
            labels = batch["label"]
            labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)

            logits: Tensor = model(input_ids)
            loss: Tensor = criterion(logits, labels_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

        # validation
        model.eval()
        preds = []
        targets = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = tokenizer(batch["verse_text"], padding=True, return_tensors="pt")\
                    .input_ids.to(device)
                labels = batch["label"]
                labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)

                logits = model(input_ids)
                preds += logits.argmax(-1).cpu().tolist()
                targets += labels

        macro = f1_score(targets, preds, average='macro')
        micro = f1_score(targets, preds, average='micro')
        print(f"[Epoch {epoch+1}] loss: {loss_sum/len(train_loader):.6f} | macro: {macro:.6f} | micro: {micro:.6f}")

    torch.save(model.cpu().state_dict(), "checkpoint.pt")