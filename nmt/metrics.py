from nltk.translate.bleu_score import corpus_bleu

def compute_bleu(trans, real, bpe_sep='@@ ', **flags):
    """
    Estimates corpora-level BLEU score of model's translations given inp and reference out
    Note: if you're serious about reporting your results, use https://pypi.org/project/sacrebleu
    """
    return corpus_bleu(
        [[ref.split()] for ref in real],
        [trans.split() for trans in trans],
        smoothing_function=lambda precisions, **kw: [p + 1.0 / p.denominator for p in precisions]
        ) * 100
