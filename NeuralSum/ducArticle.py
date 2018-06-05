

class DucArticle:
    def __init__(self, id="", folder="", sentence="", gold_summaries=[]):
        self.id = id
        self.folder = folder
        self.sentence = sentence
        self.gold_summaries = gold_summaries

    def word_count(self):
        return len(self.sentence.split())
