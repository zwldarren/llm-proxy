"""Label flywheel: turn served traffic into routing-training labels.

The pipeline is four offline stages, each a subcommand of
``python -m llm_proxy.routing.flywheel``:

1. ``sample``  — collect candidate prompts (from ``request_logs`` or an eval
   corpus) into a prompts JSONL.
2. ``label``   — for each prompt, generate one answer per tier-role model
   (weak / balanced / strong) through a proxy deployment, then pairwise-judge
   the role pairs with position swap; the resolved chain of preferences is the
   complexity label. Records stream to a labels JSONL.
3. ``train``   — train the averaged-perceptron classifier on the labeled
   examples (feature path identical to the request path), evaluate against the
   shipped model on a seeded holdout, and — only when it wins — write the
   hot-swap model artifact. Also re-fits the Platt temperature from
   (raw ensemble confidence, correct) pairs on the same holdout.

The label semantics are cost-quality, not benchmark-difficulty: a tier label
says "serving this request with the cheaper role lost nothing measurable",
which is the deployable truth (LLMRouterBench's reading of preference data),
while the authored tier of a corpus case stays available as a diagnostic.
"""

from llm_proxy.routing.flywheel.store import LabelRecord, LabelStore

__all__ = ["LabelRecord", "LabelStore"]
