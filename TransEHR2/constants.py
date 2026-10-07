import os
import torch

from dotenv import load_dotenv


if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

load_dotenv()
HF_API_TOKEN = os.getenv('HF_READ_TOKEN', None)
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'  # 5 minutes

PAD = 0
TOKENIZER_PAD_TOKEN = '[PAD]'

# Text encoder. Verified against transformers 5.17.0; see tests/test_llm_compat.py, which
# runs against whatever version is installed.
#
# bge-m3 is a plain XLMRobertaModel with no remote code: 1024-d, 8192 tokens,
# CLS-pooled, 568M parameters, 250k multilingual vocabulary. Most of that parameter count is a
# vocabulary this study does not use -- the text is English code descriptions -- which is the
# price of the model that needs no trust_remote_code on a controlled cluster.
#
# Alibaba-NLP/gte-base-en-v1.5 is smaller and better matched at 768-d and a 30k English
# vocabulary, and it is NOT usable here. Its remote code in Alibaba-NLP/new-impl registers the
# `position_ids` buffer with persistent=False and fills it via torch.arange at construction;
# v5 materializes weights into an empty model, so the buffer is never initialized and every
# forward pass dies in the embedding lookup. This fork now pins transformers>=5.0.0, so that
# is fatal rather than hypothetical -- and it would be excluded anyway, because executing
# downloaded model code is the thing to avoid on a controlled cluster.
#
# LLM_NAME = 'Alibaba-NLP/gte-base-en-v1.5'  # needs trust_remote_code; see above
# LLM_NAME = 'BAAI/bge-m3'  # the encoder ContrastiveBMMB replaces, on main
# LLM_NAME = 'meta-llama/Llama-3.1-70B'  # the decoder that replaced, masked-mean pooled
#
# This branch runs ContrastiveBMMB's A3 encoder: BioClinical ModernBERT large, contrastively
# fine-tuned on ICD-10-CA/CCI sibling substitution. 396M parameters against bge-m3's 568M, and
# 1024-d either way, so it is a dimensional drop-in. CLS-pooled, matching how it was trained --
# `TEXT_POOLING` below must not drift from ContrastiveBMMB's `POOLING`, or the lookup table and
# the encoder disagree about what a row means.
#
# A3 rather than A3w: the whitening transform lives outside the checkpoint, so loading this path
# gives unwhitened embeddings. That is the right default until the transform is folded in, and
# the attribution path takes it separately (`attribution.token_attributions`'s
# `whitening_transform`).
#
# First existing candidate wins, so one tracked value works on the cluster and on a laptop. The
# last is the fallback when none exist, which keeps the failure at `from_pretrained` -- where
# the path is named -- rather than here.
CONTRASTIVEBMMB_CANDIDATES = (
    '/uhome/pr3/projects/p60290_2/ContrastiveBMMB/checkpoints/a3',
    os.path.expanduser('~/Projects/ContrastiveBMMB/checkpoints/a3'),
)
LLM_NAME = next((p for p in CONTRASTIVEBMMB_CANDIDATES if os.path.isdir(p)),
                CONTRASTIVEBMMB_CANDIDATES[0])

# Maximum length of token sequences, and the second axis of text_tokens.npy -- so it is a
# storage decision as much as a truncation one. At section 4.5's 3.95M unique strings the
# array is 8 KB per token of width: 16 GB at 1024, 32 GB at 2048, 129 GB at bge-m3's full 8192.
#
# 2048 rather than upstream's 8192. Upstream embeds discharge summaries averaging ~2,267
# tokens; a TEXT_SUPERFEATURE here is merged ICD-10-CA and CCI descriptions, and a DAD record
# filling all 25 diagnosis and 20 intervention slots comes to roughly 3,200 characters, or
# 800-1,100 tokens. 2048 clears the worst record with headroom and costs 32 GB instead of 129.
# Raise it before extraction if the observed distribution says otherwise; nothing but
# text_tokens.npy depends on it.
MAX_TOKEN_LENGTH = 2048

# Pooling used to reduce a token sequence to one embedding. 'cls' takes position 0, which is
# what bge-m3 was contrastively trained to use; 'mean' is the masked mean over non-padding
# tokens, which is what a decoder such as Llama needs because its position 0 is only a BOS
# token.
TEXT_POOLING = 'cls'
