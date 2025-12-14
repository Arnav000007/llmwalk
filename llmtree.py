# /// script
# dependencies = [
#   "mlx-lm==0.28.4",
#   "rich==14.2.0",
#   "sortedcontainers==2.4.0",
# ]
# ///
from __future__ import annotations

import argparse
import colorsys
import time
from dataclasses import dataclass
from datetime import datetime
from threading import Thread

import mlx.core as mx
from mlx.nn import Module
from mlx_lm import load
from mlx_lm.generate import BatchGenerator, BatchStats
from mlx_lm.tokenizer_utils import TokenizerWrapper
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text
from sortedcontainers import SortedList


@dataclass
class OutputToken:
    token: int
    prob: float


@dataclass
class Branch:
    prompt: list[int]
    answer: list[OutputToken]
    probability: float = 1.0
    finish_reason: str | None = None

    def with_new_token(self, token: OutputToken) -> Branch:
        return Branch(
            prompt=self.prompt,
            answer=self.answer + [token],
            finish_reason=self.finish_reason,
            probability=self.probability * token.prob,
        )

    def tokens(self) -> list[int]:
        return self.prompt + [t.token for t in self.answer]


stats: BatchStats | None = None
active: int = 0
queued: int = 0
pruned: int = 0


def response_to_output_tokens(
    response: BatchGenerator.Response,
) -> list[OutputToken]:
    probs = mx.softmax(response.logprobs, axis=-1)
    k = min(args.topk, probs.shape[0])
    top_indices = mx.argsort(probs)[-k:][::-1]
    top_probs = mx.take(probs, top_indices)
    top_indices = top_indices.astype(mx.int64).tolist()
    top_probs = mx.reshape(top_probs, (-1,)).tolist()

    output_tokens = []
    for token_id, prob in zip(top_indices, top_probs): # type: ignore[call-arg]
        output_tokens.append(OutputToken(token=token_id, prob=prob))
    return output_tokens

class StopSignal:
    _stop = False

    def stop(self):
        self._stop = True

    @property
    def stopped(self) -> bool:
        return self._stop

class PromptTreeSearch:
    _branch_lookup: dict[int, Branch]
    branches: SortedList[Branch]
    gen: BatchGenerator
    model: Module
    tokenizer: TokenizerWrapper
    prompt: list[int]
    signal: StopSignal
    _decoded_token_cache: dict[int, str]

    tokens: int = 0
    pruned: int = 0

    _low_watermark: float = 1.0
    _start: datetime | None = None
    _end: datetime | None = None


    def __init__(self, model: Module, tokenizer: TokenizerWrapper, prompt: list[int], signal: StopSignal) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.gen = BatchGenerator(model)
        self.signal = signal
        self._decoded_token_cache = {}

        uid = self.gen.insert([prompt], max_tokens=1)[0]

        root = Branch(prompt=prompt, answer=[])
        self._branch_lookup = {}
        self._branch_lookup[uid] = root
        self.branches = SortedList(key=lambda b: -b.probability)
        self.branches.add(root)

    def decode_token(self, token_id: int) -> str:
        cached = self._decoded_token_cache.get(token_id)
        if cached is not None:
            return cached
        decoded = self.tokenizer.decode([token_id], skip_special_tokens=True)  # type: ignore[call-arg]
        self._decoded_token_cache[token_id] = decoded
        return decoded



    @property
    def active(self) -> int:
        return len(self._branch_lookup)

    @property
    def queued(self) -> int:
        return len(self.gen.unprocessed_prompts)

    @property
    def n_finished(self) -> int:
        n_finished = 0
        for branch in self.branches:
            if branch.finish_reason is not None:
                n_finished += 1
            else:
                break
        return n_finished

    def prune_branches(self):
        uids_to_remove: list[int] = []
        if self.n_finished >= args.n:
            for uid, branch in self._branch_lookup.items():
                if branch.probability < self._low_watermark:
                    self.pruned += 1
                    uids_to_remove.append(uid)
            self.gen.remove(uids_to_remove)
        for uid in uids_to_remove:
            branch = self._branch_lookup[uid]
            self.branches.remove(branch)
            del self._branch_lookup[uid]
        return uids_to_remove

    def start(self) -> Thread:
        self._start = datetime.now()

        def loop():
            responses: list[BatchGenerator.Response]
            while responses := self.gen.next():
                if self.signal.stopped:
                    break

                for r in responses:
                    self.tokens += 1
                    self.prune_branches()

                    if r.uid not in self._branch_lookup:
                        continue

                    branch = self._branch_lookup[r.uid]
                    del self._branch_lookup[r.uid]
                    self.branches.remove(branch)

                    for token in response_to_output_tokens(r):
                        new_branch = branch.with_new_token(token)

                        if new_branch.probability < args.min_probability:
                            self.pruned += 1
                            new_branch.finish_reason = "low_probability"
                            self.branches.add(new_branch)
                            continue

                        if token.token in self.tokenizer.eos_token_ids:
                            new_branch.finish_reason = "eos_token"
                            self._low_watermark = min(self._low_watermark, new_branch.probability)
                            self.branches.add(new_branch)
                            continue

                        uid = self.gen.insert([new_branch.tokens()], max_tokens=1)[0]
                        self._branch_lookup[uid] = new_branch
                        self.branches.add(new_branch)

        thread = Thread(target=loop)
        thread.start()
        return thread

def style_for_token_probability(prob: float) -> Style:
    r, g, b = colorsys.hsv_to_rgb((120 * prob) / 360.0, 0.85, 0.95)
    return Style(color=f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})")

def render_branches(walker: PromptTreeSearch) -> Table:
    table = Table(expand=True)
    table.add_column("#", justify="right", no_wrap=True, width=3)
    table.add_column("Fin", justify="center", no_wrap=True, width=5)
    table.add_column("Prob.", justify="right", no_wrap=True, width=8)
    table.add_column("Answer", ratio=1)

    for i in range(args.n):
        if i >= len(walker.branches):
            table.add_row(str(i + 1), "", "")
            continue

        branch = walker.branches[i]
        answer_text = Text()
        for tok in branch.answer:
            piece = walker.decode_token(tok.token)
            if not piece:
                continue
            answer_text.append(piece, style=style_for_token_probability(tok.prob))
        probability_text = f"{branch.probability * 100:6.2f}%"
        finished = "✓" if branch.finish_reason is not None else ""
        table.add_row(str(i + 1), finished, probability_text, answer_text)

    return table


def render_stats_bar(walker: PromptTreeSearch) -> Panel:
    elapsed = (datetime.now() - walker._start).total_seconds() if walker._start else 0.0
    tps = walker.tokens / elapsed if elapsed > 0 else 0.0
    left = f"active {walker.active}  queued {walker.queued}  pruned {walker.pruned} tps {tps:0.1f}"
    grid = Table.grid(expand=True)
    grid.add_column(ratio=1)
    grid.add_column(justify="right", no_wrap=True)
    grid.add_row(
        Text(left, overflow="ellipsis", no_wrap=True),
        Text(f"topk={args.topk}", no_wrap=True),
    )
    return Panel(grid, expand=True)


def render_view(walker: PromptTreeSearch) -> Group:
    return Group(render_branches(walker), render_stats_bar(walker))


def main() -> None:
    load_resp = load(args.model)
    model = load_resp[0]
    tokenizer = load_resp[1]

    prompt = tokenizer.apply_chat_template( # type: ignore[call-arg]
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
    )

    console = Console()

    signal = StopSignal()
    walker = PromptTreeSearch(model, tokenizer, prompt, signal)
    walker_thread = walker.start()

    try:
        with Live(console=console, transient=False) as live:
            def render():
                while not signal.stopped:
                    time.sleep(max(0.1, args.stats_interval))
                    live.update(render_view(walker))
            render_thread = Thread(target=render, daemon=True)
            render_thread.start()
            walker_thread.join()
    except KeyboardInterrupt:
        signal.stop()


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--prompt", default="What is 2+2?", help="Prompt to score")
parser.add_argument("-m", "--model", default="mlx-community/Llama-3.2-1B-Instruct-4bit")
parser.add_argument("-n", default=10, type=int, help="Number of answers to show")
parser.add_argument("--min-probability", type=float, default=0.0001)
parser.add_argument("--topk", default=50, type=int)
parser.add_argument(
    "--stats-interval",
    type=float,
    default=0.1,
    help="Seconds between live stats bar updates (<=0 disables)",
)
args = parser.parse_args()

main()
