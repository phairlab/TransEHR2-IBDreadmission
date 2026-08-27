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

# Text encoder. bge-m3 is a plain XLMRobertaModel with no remote code: 1024-d, 8192 tokens,
# CLS-pooled, 568M parameters, 250k multilingual vocabulary. Most of that parameter count is a
# vocabulary this study does not use -- the text is English code descriptions -- which is the
# price of the model that needs no trust_remote_code on a controlled cluster.
#
# Alibaba-NLP/gte-base-en-v1.5 is smaller and better matched at 768-d and a 30k English
# vocabulary, and it is NOT usable under a transformers>=5.9.0 pin. Its remote code in
# Alibaba-NLP/new-impl registers the `position_ids` buffer with persistent=False and fills it
# via torch.arange at construction; v5 materializes weights into an empty model, so the buffer
# is never initialized and every forward pass dies in the embedding lookup. This fork pins
# transformers>=4.30.0, so gte would run today -- it is excluded because executing downloaded
# model code is the thing to avoid here, not because it cannot work.
#
# LLM_NAME = 'Alibaba-NLP/gte-base-en-v1.5'  # needs trust_remote_code; see above
LLM_NAME = 'BAAI/bge-m3'
# LLM_NAME = 'meta-llama/Llama-3.1-70B'  # the decoder this replaces, masked-mean pooled

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
