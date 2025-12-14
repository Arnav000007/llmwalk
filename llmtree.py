# /// script
# dependencies = [
#   "mlx-lm>=0.28.4",
#   "rich>=14.2.0",
# ]
# ///
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Generator

import mlx.core as mx
from mlx.nn import Module
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.tokenizer_utils import TokenizerWrapper
from rich.console import Console
from rich.live import Live
from rich.table import Table


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

def walk(model: Module, tokenizer: TokenizerWrapper, prompt: list[int]) -> Generator[Branch, None, None]:
    gen = BatchGenerator(model)
    uid: int = gen.insert([prompt], max_tokens=1)[0]
    branches = { uid: Branch(prompt=prompt, answer=[]) }
    low_watermark = 1
    n_returned = 0

    def prune_branches() -> list[int]:
        uids_to_remove: list[int] = []
        if n_returned >= args.n:
            for uid, branch in branches.items():
                if branch.probability < low_watermark:
                    uids_to_remove.append(uid)
            gen.remove(uids_to_remove)
        return uids_to_remove


    global stats
    responses: list[BatchGenerator.Response]
    while responses := gen.next():
        try:
            stats = gen.stats()
        except Exception:
            pass

        for r in responses:
            if r.uid in prune_branches():
                continue

            branch = branches[r.uid]
            del branches[r.uid]

            for token in response_to_output_tokens(r):
                new_branch = branch.with_new_token(token)

                if new_branch.probability < args.min_probability:
                    new_branch.finish_reason = "low_probability"
                    yield new_branch
                    continue

                if token.token in tokenizer.eos_token_ids:
                    new_branch.finish_reason = "eos_token"
                    yield new_branch

                    n_returned += 1
                    low_watermark = min(low_watermark, new_branch.probability)

                    continue

                uid = gen.insert([new_branch.tokens()], max_tokens=1)[0]
                branches[uid] = new_branch




def render_branches(
    tokenizer: TokenizerWrapper, branches: list[Branch]
) -> Table:
    table = Table(expand=True)
    table.add_column("#", justify="right", no_wrap=True, width=3)
    table.add_column("Prob.", justify="right", no_wrap=True, width=8)
    table.add_column("Answer", ratio=1)

    for i in range(args.n):
        if i >= len(branches):
            table.add_row(str(i + 1), "", "")
            break

        branch = branches[i]
        answer_text =  tokenizer.decode(branch.answer, skip_special_tokens=True)  # type: ignore[call-arg]
        probability_text = f"{branch.probability * 100:6.2f}%"
        table.add_row(str(i + 1), probability_text, answer_text)

    return table


def main() -> None:
    load_resp = load(args.model)
    model = load_resp[0]
    tokenizer = load_resp[1]

    prompt = tokenizer.apply_chat_template( # type: ignore[call-arg]
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
    )

    branches: list[Branch] = []
    console = Console()

    with Live(
        render_branches(tokenizer, branches),
        console=console,
        refresh_per_second=8,
        transient=False,
    ) as live:
        for branch in walk(model, tokenizer, prompt):
            if branch.finish_reason == "low_probability":
                continue

            branches.append(branch)
            branches.sort(key=lambda branch: branch.probability, reverse=True)
            live.update(render_branches(tokenizer, branches))

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--prompt", default="What is 2+2?", help="Prompt to score")
parser.add_argument("-m", "--model", default="mlx-community/Llama-3.2-1B-Instruct-4bit")
parser.add_argument("-n", default=10, type=int, help="Number of answers to show")
parser.add_argument("--min-probability", type=float, default=0.0001)
parser.add_argument("--topk", default=50, type=int)
args = parser.parse_args()

main()
