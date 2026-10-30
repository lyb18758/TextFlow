"""Dependency-free regression checks; no model, torch or diffusers import is performed.

Run: python -B -m unittest discover -s textflow_longcat -p test_pipeline_safety.py -v
"""
import ast
from contextlib import nullcontext
from dataclasses import fields, is_dataclass
from pathlib import Path
import random
import re
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parent
PIPELINES = ("pipeline_longcat_attention.py", "pipeline_longcat_fms.py")


def tree(name):
    return ast.parse((ROOT / name).read_text(encoding="utf-8"), filename=name)


def method(name, method_name):
    cls = next(node for node in tree(name).body if isinstance(node, ast.ClassDef))
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)


def execute(nodes, env):
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<pipeline regression>", "exec"), env)


class Tensor:
    """Scalar arithmetic with tensor metadata, only for pipeline plumbing checks."""
    def __init__(self, value=0.0, shape=(2, 3, 4)):
        self.value, self.shape = value, shape
        self.device, self.dtype = "cpu", "float32"

    def __add__(self, other):
        return Tensor(self.value + (other.value if isinstance(other, Tensor) else other), self.shape)

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other.value if isinstance(other, Tensor) else -other)

    def __mul__(self, other):
        return Tensor(self.value * (other.value if isinstance(other, Tensor) else other), self.shape)

    __rmul__ = __mul__

    def __getitem__(self, item):
        return self

    def size(self, dim):
        return self.shape[dim]

    def repeat(self, *dims):
        return Tensor(self.value, tuple(a * b for a, b in zip(self.shape, dims)))

    def view(self, batch, length, _):
        return Tensor(self.value, (batch, length, self.shape[-1]))

    def to(self, *args, **kwargs):
        return self


class PipelineSafetyTests(unittest.TestCase):
    def resolver(self):
        node = method(PIPELINES[1], "_resolve_src_prompt")
        node.decorator_list = []
        env = {"re": re}
        execute([node], env)
        return env[node.name]

    def test_automatic_source_and_single_item_list(self):
        resolve = self.resolver()
        for prompt in ["replace 'old' with 'new'", "Replace the word 'old' with 'new'.", 'Please change the text "old" to "new".', '请将“old”替换为“new”。']:
            for target in [prompt, [prompt]]:
                with self.subTest(target=target):
                    self.assertEqual(resolve(None, target), "A photo with the word 'old'.")
        self.assertEqual(resolve(["source description"], None), "source description")

    def test_automatic_source_rejects_ambiguous_or_missing_input(self):
        resolve = self.resolver()
        for prompt in [None, [], ["a", "b"], "change the color", "write 'new'", "replace ' ' with 'new'", "replace 'old' with ' '", "replace 'old' with 'new' and change color", "replace 'old\" with 'new'"]:
            with self.subTest(prompt=prompt), self.assertRaisesRegex(ValueError, "Cannot safely infer src_prompt"):
                resolve(None, prompt)
        for source in ["", " ", [], ["a", "b"], [None], 1]:
            with self.subTest(source=source), self.assertRaises(ValueError):
                resolve(source, "target")

    def test_source_encoding_does_not_receive_target_embeddings_or_ids(self):
        call = method(PIPELINES[1], "__call__")
        block = next(node for node in call.body if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "use_flowedit_enhancement" and any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Tuple) for t in n.targets) for n in node.body))
        calls = []
        def encode_prompt(**kwargs):
            calls.append(kwargs)
            return "source embeddings", "source ids"
        env = dict(self=SimpleNamespace(encode_prompt=encode_prompt), use_flowedit_enhancement=True, src_prompt="source description", prompt_image="image", num_images_per_prompt=2, prompt_embeds="target embeddings", text_ids="target ids", logger=SimpleNamespace(info=lambda *args: None), flowedit_start_step=0, flowedit_end_step=24)
        execute([block], env)
        self.assertEqual(env["prompt_embeds"], "target embeddings")
        self.assertEqual(env["text_ids"], "target ids")
        self.assertEqual(env["src_text_ids"], "source ids")
        self.assertEqual(calls, [dict(prompt="source description", image="image", num_images_per_prompt=2)])
        encode = method(PIPELINES[1], "_encode_prompt")
        text_encoder_call = next(n for n in ast.walk(encode) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "text_encoder")
        self.assertTrue(any(k.arg == "attention_mask" and isinstance(k.value, ast.Name) and k.value.id == "attention_mask" for k in text_encoder_call.keywords))

    def test_generator_and_source_ids_reach_velocity_computation(self):
        node = method(PIPELINES[1], "_compute_velocity_delta")
        env = {"torch": SimpleNamespace(Tensor=Tensor, zeros_like=lambda x: Tensor(shape=x.shape), tensor=lambda *a, **k: Tensor(), float32="float32", no_grad=nullcontext)}
        noise_generators = []
        def randn_tensor(shape, generator, **kwargs):
            noise_generators.append(generator)
            generators = generator if isinstance(generator, list) else [generator]
            return Tensor(sum(item.random() for item in generators) / len(generators), shape)
        env["randn_tensor"] = randn_tensor
        execute([node], env)
        calls = []
        def transformer(**kwargs):
            calls.append(kwargs)
            return [kwargs["hidden_states"] * kwargs["encoder_hidden_states"]]
        scheduler = SimpleNamespace(_init_step_index=lambda t: None, sigmas=[0.5], step_index=0)
        pipe = SimpleNamespace(scheduler=scheduler, transformer=transformer)
        def run(seed):
            return env[node.name](pipe, Tensor(2), Tensor(1), None, 2, 3, None, None, "target ids", "source ids", "image ids", 500.0, n_avg=2, generator=random.Random(seed)).value
        self.assertEqual(run(7), run(7))
        self.assertNotEqual(run(7), run(8))
        self.assertEqual(len(noise_generators), 8)
        generators = [random.Random(7), random.Random(8)]
        env[node.name](pipe, Tensor(2), Tensor(1), None, 2, 3, None, None, "target ids", "source ids", "image ids", 500.0, generator=generators)
        self.assertIs(noise_generators[-1], generators)
        pipeline_call = next(n for n in ast.walk(method(PIPELINES[1], "__call__")) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "_compute_velocity_delta")
        self.assertTrue(any(k.arg == "generator" and isinstance(k.value, ast.Name) and k.value.id == "generator" for k in pipeline_call.keywords))
        for source, target in zip(calls[::2], calls[1::2]):
            self.assertEqual(source["txt_ids"], "source ids")
            self.assertEqual(target["txt_ids"], "target ids")
        noise_call = next(n for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "randn_tensor")
        self.assertTrue(any(k.arg == "generator" and isinstance(k.value, ast.Name) and k.value.id == "generator" for k in noise_call.keywords))

    def test_attention_state_cleared_and_unsupported_scheduler_rejected(self):
        for name in PIPELINES:
            call = method(name, "__call__")
            block = next(n for n in call.body if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "use_att" and any(isinstance(t, ast.Call) and isinstance(t.func, ast.Name) and t.func.id == "callable" for t in ast.walk(n)))
            scheduler = SimpleNamespace(attn_map="stale")
            scheduler.set_attn_map = lambda value: setattr(scheduler, "attn_map", value)
            execute([block], dict(self=SimpleNamespace(scheduler=scheduler), use_att=False))
            self.assertIsNone(scheduler.attn_map)
            execute([block], dict(self=SimpleNamespace(scheduler=object()), use_att=False))
            with self.assertRaisesRegex(ValueError, "set_attn_map"):
                execute([block], dict(self=SimpleNamespace(scheduler=object()), use_att=True))

    def test_scheduler_step_forwards_single_and_list_generators(self):
        for name in PIPELINES:
            step = next(n for n in ast.walk(method(name, "__call__")) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Subscript) and isinstance(n.value.value, ast.Call) and isinstance(n.value.value.func, ast.Attribute) and n.value.value.func.attr == "step")
            for generator in [object(), [object(), object()]]:
                seen = []
                scheduler = SimpleNamespace(step=lambda *args, **kwargs: seen.append(kwargs) or ["next latent"])
                env = dict(self=SimpleNamespace(scheduler=scheduler), noise_pred="noise", t=1, latents="latent", generator=generator)
                execute([step], env)
                self.assertIs(seen[0]["generator"], generator)
                self.assertEqual(env["latents"], "next latent")

    def test_embedding_batch_expansion_and_precomputed_guard(self):
        for name in PIPELINES:
            node = method(name, "encode_prompt")
            env = {"prepare_pos_ids": lambda **kwargs: Tensor(shape=(kwargs["num_token"], 3))}
            execute([node], env)
            pipe = SimpleNamespace(device="cpu", _encode_prompt=lambda *args, **kwargs: Tensor(shape=(1, 7, 4)))
            embeddings, ids = env[node.name](pipe, ["source"], num_images_per_prompt=2)
            self.assertEqual(embeddings.shape, (2, 7, 4))
            self.assertEqual(ids.shape, (7, 3))
            embeddings, _ = env[node.name](pipe, prompt_embeds=Tensor(shape=(1, 5, 4)), num_images_per_prompt=2)
            self.assertEqual(embeddings.shape, (2, 5, 4))
            with self.assertRaisesRegex(ValueError, "precomputed"):
                env[node.name](pipe, prompt_embeds=Tensor(), get_text_to_render=True)

    def test_output_dataclass_accepts_optional_attention(self):
        nodes = [node for node in tree("outputs.py").body if not isinstance(node, ast.ImportFrom) or node.module == "dataclasses"]
        env = {"BaseOutput": type("BaseOutput", (), {})}
        execute(nodes, env)
        output = env["Transformer2DModelOutput"]
        self.assertTrue(is_dataclass(output))
        self.assertEqual([field.name for field in fields(output)], ["sample", "attn_map"])
        self.assertIsNone(output(sample="sample").attn_map)
        self.assertEqual(output(sample="sample", attn_map=["attention"]).attn_map, ["attention"])


if __name__ == "__main__":
    unittest.main()
