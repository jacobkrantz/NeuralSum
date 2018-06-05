

class DucArticle:
    def __init__(self, id="", folder="", sentence="", gold_summaries=[]):
        self.id = id
        self.folder = folder
        self.sentence = sentence
        self.gold_summaries = gold_summaries

    def word_count(self):
        return len(self.sentence.split())

    def __str__(self):
        s = "Article ID: " + self.id + " Folder: " + self.folder
        s += '\n' + "Sentence: " + self.sentence
        s += '\n' + "Summaries: "
        for i, summary in enumerate(self.gold_summaries):
            s += '\n' + str(i+1) + ": " + summary
        return s
