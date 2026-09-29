"""The transformers API surface, against whatever version is installed.

Every other test stubs the tokenizer and the embedder, which is right for
what they check -- and it is why a breaking change in ``from_pretrained``
sailed through a green suite. ``GradientTraceableLLM`` was calling
``AutoModel.from_pretrained(dtype=...)``, an argument that did not exist
before transformers 4.56 and that arrives at the model constructor as an
unexpected keyword rather than a warning. The whole text path died at load
time on 4.32 through 4.55, and nothing said so.

So these load a real model through the real API. It is a two-layer
XLM-RoBERTa -- bge-m3's architecture -- built and saved here, because a
compatibility test that needs a 568M-parameter download is a test nobody
runs.
"""

import json
import pytest
import torch

from transformers import (
    AutoModel, AutoTokenizer, XLMRobertaConfig, XLMRobertaModel,
)

from TransEHR2.modules import GradientTraceableLLM, dtype_kwarg


HIDDEN = 32
VOCAB = 1000


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    """A miniature XLM-RoBERTa and tokenizer, saved like a HF repository."""
    path = tmp_path_factory.mktemp("tiny_xlmr")

    XLMRobertaModel(XLMRobertaConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=64,
        max_position_embeddings=64, type_vocab_size=1,
    )).save_pretrained(path)

    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
    for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz ", start=4):
        vocab[ch] = i
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)])
    PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token="<s>", eos_token="</s>",
        unk_token="<unk>", pad_token="<pad>",
        model_input_names=["input_ids", "attention_mask"],
    ).save_pretrained(path)

    config_path = path / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["tokenizer_class"] = "PreTrainedTokenizerFast"
    config_path.write_text(json.dumps(config))

    return str(path)


# ------------------------------------------------------------ the translator

def test_no_dtype_passes_no_keyword():
    """Passing ``dtype=None`` by name is what broke: it is an argument."""
    assert dtype_kwarg(None) == {}


def test_the_dtype_keyword_is_named_for_the_installed_version():
    import transformers
    major, minor = (int(p) for p in transformers.__version__.split('.')[:2])
    expected = 'dtype' if (major, minor) >= (4, 56) else 'torch_dtype'
    assert list(dtype_kwarg(torch.bfloat16)) == [expected]


def test_the_translated_keyword_is_actually_honoured(tiny_model):
    """The name has to be right, not merely accepted -- a kwarg the loader
    swallows into the config would leave the weights in float32."""
    model = AutoModel.from_pretrained(tiny_model,
                                      **dtype_kwarg(torch.bfloat16))
    assert next(model.parameters()).dtype is torch.bfloat16


# ---------------------------------------------------------------- the wrapper

def test_the_wrapper_loads_at_the_dtype_embed_py_asks_for(tiny_model):
    """``scripts/embed.py`` constructs it with ``dtype=torch.bfloat16``."""
    llm = GradientTraceableLLM(tiny_model, max_length=16,
                               use_gradient_checkpointing=False,
                               dtype=torch.bfloat16)
    assert next(llm.model.parameters()).dtype is torch.bfloat16


def test_the_wrapper_freezes_the_encoder(tiny_model):
    llm = GradientTraceableLLM(tiny_model, max_length=16,
                               use_gradient_checkpointing=False)
    assert not any(p.requires_grad for p in llm.model.parameters())


def test_gradient_checkpointing_can_be_enabled(tiny_model):
    llm = GradientTraceableLLM(tiny_model, max_length=16,
                               use_gradient_checkpointing=True)
    assert llm.model.is_gradient_checkpointing


def test_an_existing_pad_token_is_left_alone(tiny_model):
    """bge-m3 ships ``<pad>``; adding a second would grow the vocabulary by
    a row the checkpoint never trained and move padding off the token the
    model learned to ignore."""
    tokenizer = AutoTokenizer.from_pretrained(tiny_model)
    before = len(tokenizer)
    llm = GradientTraceableLLM(tiny_model, max_length=16,
                               use_gradient_checkpointing=False)
    assert llm.tokenizer.pad_token == "<pad>"
    assert len(llm.tokenizer) == before


def test_the_embedding_matrix_is_resized_to_the_tokenizer(tiny_model):
    """The fixture's config vocabulary is deliberately wider than the
    tokenizer's, which is the condition the resize exists for."""
    llm = GradientTraceableLLM(tiny_model, max_length=16,
                               use_gradient_checkpointing=False)
    assert llm.model.get_input_embeddings().weight.size(0) == \
        len(llm.tokenizer)


# ----------------------------------------------------------------- the pooling

def _encode(llm, strings, length=16):
    return llm.tokenizer(strings, padding='max_length', truncation=True,
                         max_length=length, return_tensors='pt')


def test_cls_pooling_returns_one_vector_per_string(tiny_model):
    llm = GradientTraceableLLM(tiny_model, max_length=16,
                               use_gradient_checkpointing=False,
                               pooling='cls')
    encoded = _encode(llm, ["hello there", "a"])
    out = llm(encoded['input_ids'],
              attention_mask=encoded['attention_mask'])
    assert out.shape == (2, HIDDEN)
    assert torch.isfinite(out).all()


def test_mean_pooling_excludes_padding(tiny_model):
    """Without the mask the padding rows join the mean, which is the bug
    the mask branch exists to avoid."""
    llm = GradientTraceableLLM(tiny_model, max_length=16,
                               use_gradient_checkpointing=False,
                               pooling='mean')
    encoded = _encode(llm, ["hello there"])
    masked = llm(encoded['input_ids'],
                 attention_mask=encoded['attention_mask'])
    unmasked = llm(encoded['input_ids'], attention_mask=None)
    assert not torch.allclose(masked, unmasked)


def test_an_unknown_pooling_is_refused(tiny_model):
    with pytest.raises(ValueError, match='pooling'):
        GradientTraceableLLM(tiny_model, max_length=16,
                             use_gradient_checkpointing=False,
                             pooling='max')


def test_gradients_reach_the_embedding_matrix(tiny_model):
    """Token-level attribution needs this; the encoder is frozen, so the
    gradient has to be re-enabled on the embeddings deliberately."""
    llm = GradientTraceableLLM(tiny_model, max_length=16,
                               use_gradient_checkpointing=False)
    llm.model.get_input_embeddings().weight.requires_grad_(True)
    encoded = _encode(llm, ["hello"])
    out = llm(encoded['input_ids'],
              attention_mask=encoded['attention_mask'], trace_grads=True)
    out.sum().backward()
    assert llm.embedding_gradients is not None
    assert llm.embedding_gradients.shape == (1, HIDDEN)


def test_no_gradient_is_traced_by_default(tiny_model):
    llm = GradientTraceableLLM(tiny_model, max_length=16,
                               use_gradient_checkpointing=False)
    encoded = _encode(llm, ["hello"])
    out = llm(encoded['input_ids'],
              attention_mask=encoded['attention_mask'])
    assert not out.requires_grad


# ------------------------------------------------- the preprocessing tokenizer

def test_the_tokenizer_right_pads(tiny_model):
    """CLS pooling reads position 0, so left padding would make every short
    string's embedding the padding token's."""
    tokenizer = AutoTokenizer.from_pretrained(tiny_model,
                                              local_files_only=True)
    encoded = tokenizer(["abc", "de"], padding='max_length', truncation=True,
                        max_length=8, return_tensors='np')
    assert encoded['input_ids'].shape == (2, 8)
    assert encoded['attention_mask'][1][0] == 1
    assert encoded['attention_mask'][1][-1] == 0
