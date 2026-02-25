import torch
import torch.nn.functional as F
from typing import List, Tuple
import kenlm

# Example Usage (to incorporate into pipeline) #
# LM_PATH = "phoneme_lm.arpa"
# phoneme_map = create_phoneme_map()
# lm = PhonemeLM(LM_PATH, phoneme_map=phoneme_map)

PHONE_DEF = [
    'AA','AE','AH','AO','AW',
    'AY','B','CH','D','DH',
    'EH','ER','EY','F','G',
    'HH','IH','IY','JH','K',
    'L','M','N','NG','OW',
    'OY','P','R','S','SH',
    'T','TH','UH','UW','V',
    'W','Y','Z','ZH'
]

PHONE_DEF_SIL = PHONE_DEF + ['SIL']
index2phone = {i:p for i,p in enumerate(PHONE_DEF_SIL)}
phone2index = {p:i for i,p in enumerate(PHONE_DEF_SIL)}

# ---------------------------
# LANGUAGE MODEL
# ---------------------------
class PhonemeLM:
    def __init__(self, lm_path: str):
        self.model = kenlm.Model(lm_path)

    def score(self, tokens: List[str]) -> float:
        line = " ".join(tokens)
        return self.model.score(line, bos=False, eos=False)

# ---------------------------
# BEAM SEARCH
# ---------------------------
def lm_beam_search(
    logits: torch.Tensor,      
    lm: PhonemeLM,
    beam_size: int = 10,
    lm_weight: float = 0.4,
    length_penalty: float = 0.0
):

    T, V = logits.size()
    log_probs = F.log_softmax(logits, dim=-1)

    beam = [([], 0.0, 0.0)]

    for t in range(T):
        new_beam = []
        lp_t = log_probs[t]

        for tokens, a_score, lm_score in beam:
            for v in range(V):
                new_tok = tokens + [index2phone[v]]
                new_a = a_score + lp_t[v].item()
                new_lm = lm.score(new_tok)

                fused = new_a + lm_weight * new_lm

                new_beam.append((new_tok, new_a, new_lm, fused))

        new_beam = sorted(new_beam, key=lambda x: x[3], reverse=True)[:beam_size]
        beam = [(toks, a, l) for (toks, a, l, _) in new_beam]

    beam = sorted(beam, key=lambda x: x[1] + lm_weight * x[2], reverse=True)

    return beam[0][0]   
