from datasets import load_dataset


def load_corpus() -> list[str]:
    corpus: list[str] = []
    dataset = load_dataset("google-research-datasets/poem_sentiment")

    for split in ["train", "validation", "test"]:
        for example in dataset[split]:
            text = example["verse_text"]
            if isinstance(text, str) and text.strip():
                corpus.append(text.strip())

    return corpus