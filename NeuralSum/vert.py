
import datetime
import json
from itertools import starmap
import numpy as np
from rouge import Rouge # https://github.com/pltrdy/rouge
from InferSent import InferSent
from word_mover_distance import WordMoverDistance
import sys


reload(sys)
sys.setdefaultencoding('utf8')

class Vert(object):
    """
        Performs automatic evaluation of summaries using the proposed
            VERT metric. Generates score reports and can output to file.
        Can be memory intensive.
    """

    def __init__(self, save_memory=True):
        """
        save_memory (bool): if True, removes vocabularies after testing.
            If false, saves vocabularies in memory of the class object.
            Does not change the VERT scores.
        """
        self.infersent = None
        self.wmd = None
        self.rouge = None
        self.save_mem = save_memory

    def score_from_articles(articles, rouge_type='recall', verbose=False):
        """
        Args:
            articles (list<DucArticle>)
            rouge_type (string): can be 'recall', 'precision', or 'f-measure'
            verbose (bool): if True prints status updates.
        Returns:
            dict of average score results. keys:
                [
                  'rouge-1','rouge-2','rouge-l',
                  'cos_sim','wm_dist','vert_score',
                  'test_type','num_tested',
                  'avg_hyp_word_cnt','avg_hyp_word_cnt'
                ]
        """
        hyps, refs = [],[]
        for art in articles:
            hyps += [art.generated_summary for i in range(len(article.gold_summaries))]
            refs += art.gold_summaries
        return self.score(hyps, refs, rouge_type=rouge_type, verbose=verbose)

    def score(self, hyps, refs, rouge_type='recall', verbose=False):
        """
        Args:
            hyps (list<string>): generated summary list to score
            refs (list<string>): reference summary list to test against
            rouge_type (string): can be 'recall', 'precision', or 'f-measure'
            verbose (bool): if True prints status updates.
        Returns:
            dict of average score results. keys:
                [
                  'rouge-1','rouge-2','rouge-l',
                  'cos_sim','wm_dist','vert_score',
                  'test_type','num_tested',
                  'avg_hyp_word_cnt','avg_hyp_word_cnt'
                ]
        """
        self._verify_input(hyps, refs)

        # calculate average InferSent cosine similarity of sentence vectors
        if self.infersent is None:
            self.infersent = InferSent()

        cos_sim, all_cos_sims = self.infersent.get_avg_similarity(hyps, refs)
        if self.save_mem:
            del self.infersent
            self.infersent = None
        if verbose:
            print('InferSent scores calculated.')

        # calculate average word mover's distance
        if self.wmd is None:
            del self.wmd
            self.wmd = WordMoverDistance()

        wmd_score, all_wmd_scores = self.wmd.get_avg_wmd(hyps, refs)
        if self.save_mem:
            self.wmd = None
        if verbose:
            print('Word Mover\'s Distance scores calculated.')

        # calculate VERT score
        vert_score = self._calc_avg_vert_score(all_cos_sims, all_wmd_scores)
        # alternate way(same result):
        # vert_score = self._calc_vert_score_post(cos_sim, wmd_score)

        # calculate rouge scores
        if self.rouge is None:
            del self.rouge
            self.rouge = Rouge()

        r_scores = self.rouge.get_scores(hyps, refs, avg=True)
        rouge_1 = r_scores['rouge-1'][rouge_type[0]] * 100
        rouge_2 = r_scores['rouge-2'][rouge_type[0]] * 100
        rouge_l = r_scores['rouge-l'][rouge_type[0]] * 100
        if self.save_mem:
            self.rouge = None
        if verbose:
            print('ROUGE ' + rouge_type + ' scores calculated.')

        # compile and return scores
        scores = {
            'rouge-1':"{0:.3f}".format(rouge_1),
            'rouge-2':"{0:.3f}".format(rouge_2),
            'rouge-l':"{0:.3f}".format(rouge_l),
            'cos_sim':"{0:.4f}".format(cos_sim),
            'wm_dist':"{0:.3f}".format(wmd_score),
            'vert_score':"{0:.3f}".format(vert_score),
            'avg_hyp_word_cnt':"{0:.3f}".format(self._get_avg_word_count(hyps)),
            'avg_ref_word_cnt':"{0:.3f}".format(self._get_avg_word_count(refs)),
            'test_type': rouge_type,
            'num_tested':str(len(hyps)),
        }
        if verbose:
            self.display_scores(scores)
        return scores

    @classmethod
    def output_report(self, scores, filepath):
        """
        Outputs the given scores dictionary to a json file in
            the rouge reports folder
        Args:
            scores (dict): VERT scores
            filename (string): location to output score report
        Returns:
            None
        """
        dt = str(datetime.datetime.now().isoformat())[:-5]
        name = "vert_scores_" + dt + ".json"
        with open(filepath + name, 'w') as f:
            json.dump(scores, f, indent=2, sort_keys=True)

    @classmethod
    def display_scores(self, scores):
        print('-----------------------------')
        print('         VERT scores         ')
        print('-----------------------------')
        for k, v in scores.iteritems():
            tab = ':\t\t' if len(k) < 15 else ':\t'
            print(k + tab + str(v))

    def _verify_input(self, hyps, refs):
        """
        Makes sures the data is in proper format for score tests.
        """
        assert(len(hyps) == len(refs))
        bad_ids = []
        for i in range(len(hyps)):
            if (hyps[i] == '') or (refs[i] == ''):
                bad_ids.append(i)
        bad_ids.sort(reverse=True) # reverse so we can pop by id
        for bad_id in bad_ids:
            hyps.pop(bad_id)
            refs.pop(bad_id)
        if len(bad_ids) > 0:
            print("WARNING: Removed " + str(len(bad_ids)) + " testing points: null values.")

        assert(len(hyps) == len(refs))
        for i, sent in enumerate(hyps):
            assert(len(sent) > 0)
        for sent in refs:
            assert(len(sent) > 0)

    def _get_avg_word_count(self, sentences):
        """
        From a list of sentence strings, calculate the average number of words.
        Returns:
            float
        """
        return np.mean(list(map(lambda sen: len(sen.split()), sentences)))

    def _calc_avg_vert_score(self, sim_scores, dis_scores):
        assert(len(sim_scores) == len(dis_scores))
        return np.mean(list(starmap(
            lambda s,d: self._calc_vert_score_post(s,d),
            zip(sim_scores, dis_scores)
        )))

    def _calc_vert_score_post(self, similarity, dissimilarity):
        return np.tanh(float(similarity) / dissimilarity**(1./3.))
