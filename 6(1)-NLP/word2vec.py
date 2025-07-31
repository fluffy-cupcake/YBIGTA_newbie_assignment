import torch
from torch import nn, Tensor, LongTensor
from torch.optim import Adam

from transformers import PreTrainedTokenizer

from typing import Literal

# 구현하세요!


class Word2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        window_size: int,
        method: Literal["cbow", "skipgram"]
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.weight = nn.Linear(d_model, vocab_size, bias=False)
        self.window_size = window_size
        self.method = method
        self.vocab_size = vocab_size
        pass

    def embeddings_weight(self) -> Tensor:
        return self.embeddings.weight.detach()

    def fit(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        lr: float,
        num_epochs: int
    ) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr)
        # 구현하세요!
        # tokenize corpus → LongTensor 시퀀스 만들기
        tokenized = tokenizer(corpus, padding=False, truncation=False)["input_ids"]

        # padding token id 확인
        pad_token_id = tokenizer.pad_token_id
        tokenized = [
            [tok for tok in sent if tok != pad_token_id]
            for sent in tokenized
        ]

        for epoch in range(num_epochs):
            total_loss = 0.0
            for sent in tokenized:
                if len(sent) < self.window_size * 2 + 1:
                    continue  # skip too short sentences

                if self.method == "cbow":
                    loss = self._train_cbow(sent, criterion)
                elif self.method == "skipgram":
                    loss = self._train_skipgram(sent, criterion)
                else:
                    raise ValueError(f"Unknown method: {self.method}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"[Epoch {epoch+1}] loss: {total_loss:.4f}")
        pass

    def _train_cbow(
        self,
        sent: list[int],
        criterion: nn.CrossEntropyLoss
    ) -> Tensor:
        loss = torch.tensor(0.0, device=self.embeddings.weight.device)
        center_offset = self.window_size

        for i in range(center_offset, len(sent) - center_offset):
            center = sent[i]
            context = sent[i - center_offset:i] + sent[i + 1:i + center_offset + 1]
            context_tensor = torch.tensor(context, device=self.embeddings.weight.device)

            # context vector: 평균값 (cbow)
            context_embed = self.embeddings(context_tensor)  # (window*2, d_model)
            context_mean = context_embed.mean(dim=0, keepdim=True)  # (1, d_model)

            # projection
            logits = self.weight(context_mean)  # (1, vocab_size)
            target = torch.tensor([center], device=self.embeddings.weight.device)

            loss += criterion(logits, target)

        return loss

    def _train_skipgram(
        self,
        sent: list[int],
        criterion: nn.CrossEntropyLoss
    ) -> Tensor:
        loss = torch.tensor(0.0, device=self.embeddings.weight.device)
        center_offset = self.window_size

        for i in range(center_offset, len(sent) - center_offset):
            center = sent[i]
            context = sent[i - center_offset:i] + sent[i + 1:i + center_offset + 1]

            center_tensor = torch.tensor([center], device=self.embeddings.weight.device)
            center_embed = self.embeddings(center_tensor)  # (1, d_model)

            for context_word in context:
                context_tensor = torch.tensor([context_word], device=self.embeddings.weight.device)
                logits = self.weight(center_embed)  # (1, vocab_size)
                loss += criterion(logits, context_tensor)

        return loss