# app/bigram_model.py
import random
from collections import defaultdict

class BigramModel:
    def __init__(self, corpus):
        self.bigram_probs = self._build_bigram_model(corpus)

    def _tokenize(self, text):
        return text.lower().split()

    def _build_bigram_model(self, corpus):
        bigram_counts = defaultdict(lambda: defaultdict(int))
        for sentence in corpus:
            tokens = self._tokenize(sentence)
            for i in range(len(tokens) - 1):
                bigram_counts[tokens[i]][tokens[i + 1]] += 1

        bigram_probs = {
            word: {next_word: count / sum(next_words.values())
                   for next_word, count in next_words.items()}
            for word, next_words in bigram_counts.items()
        }
        return bigram_probs

    def generate_text(self, start_word: str, length: int = 10):
        start_word = start_word.lower()
        if start_word not in self.bigram_probs:
            return f"'{start_word}' not found in corpus."

        current_word = start_word
        result = [current_word]

        for _ in range(length - 1):
            next_words = self.bigram_probs.get(current_word, None)
            if not next_words:
                break
            next_word = random.choices(
                list(next_words.keys()),
                weights=next_words.values()
            )[0]
            result.append(next_word)
            current_word = next_word
        return " ".join(result)
