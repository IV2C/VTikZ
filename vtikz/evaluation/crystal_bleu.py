"""The implementation of the CrystalBleu metric (Papineni et al., 2002)."""

import logging
from importlib import import_module
from typing import List, Sequence, Optional, Dict, Any

from sacrebleu import BLEU
from sacrebleu.metrics.bleu import BLEUSignature

from sacrebleu.metrics.helpers import extract_all_word_ngrams

sacrelogger = logging.getLogger("sacrebleu")

# The default for the maximum n-gram order when computing precisions
MAX_NGRAM_ORDER = 8

_TOKENIZERS = {
    "none": "tokenizer_none.NoneTokenizer",
    "zh": "tokenizer_zh.TokenizerZh",
    "13a": "tokenizer_13a.Tokenizer13a",
    "intl": "tokenizer_intl.TokenizerV14International",
    "char": "tokenizer_char.TokenizerChar",
    "ja-mecab": "tokenizer_ja_mecab.TokenizerJaMecab",
    "ko-mecab": "tokenizer_ko_mecab.TokenizerKoMecab",
    "spm": "tokenizer_spm.TokenizerSPM",
    "flores101": "tokenizer_spm.Flores101Tokenizer",
    "flores200": "tokenizer_spm.Flores200Tokenizer",
}


def _get_tokenizer(name: str):
    """Dynamically import tokenizer as importing all is slow."""
    module_name, class_name = _TOKENIZERS[name].rsplit(".", 1)
    return getattr(import_module(f".tokenizers.{module_name}", "sacrebleu"), class_name)


class CrystalBLEU(BLEU):
    """Computes the CrystalBLEU metric given hypotheses and references.

    :param lowercase: If True, lowercased CrystalBLEU is computed.
    :param force: Ignore data that looks already tokenized.
    :param tokenize: The tokenizer to use. If None, defaults to language-specific tokenizers with '13a' as the fallback default.
    :param smooth_method: The smoothing method to use ('floor', 'add-k', 'exp' or 'none').
    :param smooth_value: The smoothing value for `floor` and `add-k` methods. `None` falls back to default value.
    :param max_ngram_order: If given, it overrides the maximum n-gram order (default: 4) when computing precisions.
    :param effective_order: If `True`, stop including n-gram orders for which precision is 0. This should be
        `True`, if sentence-level CrystalBLEU will be computed.
    :param trg_lang: An optional language code to raise potential tokenizer warnings.
    :param references: A sequence of reference documents with document being
    defined as a sequence of reference strings. If given, the reference n-grams
    and lengths will be pre-computed and cached for faster CrystalBLEU computation
    across many systems.
    """

    SMOOTH_DEFAULTS: Dict[str, Optional[float]] = {
        # The defaults for `floor` and `add-k` are obtained from the following paper
        # A Systematic Comparison of Smoothing Techniques for Sentence-Level CrystalBLEU
        # Boxing Chen and Colin Cherry
        # http://aclweb.org/anthology/W14-3346
        "none": None,  # No value is required
        "floor": 0.1,
        "add-k": 1,
        "exp": None,  # No value is required
    }

    TOKENIZERS = _TOKENIZERS.keys()

    # mteval-v13a.pl tokenizer unless Chinese or Japanese is provided
    TOKENIZER_DEFAULT = "13a"

    # Some language specific mappings to use if `trg_lang` is given
    # and the tokenizer is not explicitly specified
    _TOKENIZER_MAP = {
        "zh": "zh",
        "ja": "ja-mecab",
        "ko": "ko-mecab",
    }

    _SIGNATURE_TYPE = BLEUSignature

    def __init__(
        self,
        lowercase: bool = False,
        force: bool = False,
        tokenize: Optional[str] = None,
        smooth_method: str = "exp",
        smooth_value: Optional[float] = None,
        max_ngram_order: int = MAX_NGRAM_ORDER,
        effective_order: bool = False,
        trg_lang: str = "",
        references: Optional[Sequence[Sequence[str]]] = None,
        full_corpus: Sequence[str] = [],  # list of strs, part of the corpus
        k: int = 50,  # number of ngrams to put in the skip ngrams array
    ):
        """`CrystalBLEU` initializer."""

        super().__init__(
            lowercase,
            force,
            tokenize,
            smooth_method,
            smooth_value,
            max_ngram_order,
            effective_order,
            trg_lang,
            references,
        )

        corpus = " ".join([self._preprocess_segment(x) for x in full_corpus])

        corpus_ngrams, _ = extract_all_word_ngrams(self.tokenizer(corpus), 1, 3)
        self.most_occuring_ngrams = [ngram[0] for ngram in corpus_ngrams.most_common(k)]

    def _extract_reference_info(self, refs: Sequence[str]) -> Dict[str, Any]:
        """Given a list of reference segments, extract the n-grams and reference lengths.
        The latter will be useful when comparing hypothesis and reference lengths for CrystalBLEU.

        :param refs: A sequence of strings.
        :return: A dictionary that will be passed to `_compute_segment_statistics()`
        through keyword arguments.
        """
        ngrams = None
        ref_lens = []

        for ref in refs:
            # extract n-grams for this ref
            this_ngrams, ref_len = extract_all_word_ngrams(ref, 1, self.max_ngram_order)
            ref_lens.append(ref_len)

            this_ngrams = {
                ref_ngram: ref_count
                for ref_ngram, ref_count in this_ngrams.items()
                if ref_ngram not in self.most_occuring_ngrams
            }  # CrystalBleu: removal of most occuring n-grams in reference

            if ngrams is None:
                # Set it directly for first set of refs
                ngrams = this_ngrams
            else:
                # Merge counts across multiple references
                # The below loop is faster than `ngrams |= this_ngrams`
                for ngram, count in this_ngrams.items():
                    ngrams[ngram] = max(ngrams[ngram], count)

        return {"ref_ngrams": ngrams, "ref_lens": ref_lens}

    def _compute_segment_statistics(
        self, hypothesis: str, ref_kwargs: Dict
    ) -> List[int]:
        """Given a (pre-processed) hypothesis sentence and already computed
        reference n-grams & lengths, returns the best match statistics across the
        references.

        :param hypothesis: Hypothesis sentence.
        :param ref_kwargs: A dictionary with `refs_ngrams`and `ref_lens` keys
        that denote the counter containing all n-gram counts and reference lengths,
        respectively.
        :return: A list of integers with match statistics.
        """

        ref_ngrams, ref_lens = ref_kwargs["ref_ngrams"], ref_kwargs["ref_lens"]

        # Extract n-grams for the hypothesis
        hyp_ngrams, hyp_len = extract_all_word_ngrams(
            hypothesis, 1, self.max_ngram_order
        )

        hyp_ngrams = {
            hyp_ngram: hyp_count
            for hyp_ngram, hyp_count in hyp_ngrams.items()
            if hyp_ngram not in self.most_occuring_ngrams
        }  # CrystalBleu: removal of most occuring n-grams in hypothesis

        ref_len = self._get_closest_ref_len(hyp_len, ref_lens)

        # Count the stats
        # Although counter has its internal & and | operators, this is faster
        correct = [0 for i in range(self.max_ngram_order)]
        total = correct[:]
        for hyp_ngram, hyp_count in hyp_ngrams.items():
            # n-gram order
            n = len(hyp_ngram) - 1
            # count hypothesis n-grams
            total[n] += hyp_count
            # count matched n-grams
            if hyp_ngram in ref_ngrams:
                correct[n] += min(hyp_count, ref_ngrams[hyp_ngram])

        # Return a flattened list for efficient computation
        return [hyp_len, ref_len] + correct + total
