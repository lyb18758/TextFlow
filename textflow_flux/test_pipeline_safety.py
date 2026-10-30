"""Dependency-free regression checks; parse local sources without importing ML code.

Run: python -m unittest discover -s textflow_flux -p test_pipeline_safety.py
"""

import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parent
PIPELINES = (
    "pipeline_flux_kontext_deferred_handoff.py",
    "pipeline_flux_kontext_stepwise_injection.py",
)
SCHEDULER = "scheduling_attention_overshoot.py"


def calls(node, name):
    return [
        item for item in ast.walk(node)
        if isinstance(item, ast.Call)
        and ((isinstance(item.func, ast.Name) and item.func.id == name)
             or (isinstance(item.func, ast.Attribute) and item.func.attr == name))
    ]


def execute(nodes, **env):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<contract>", "exec"), env)


class SchedulerStub:
    def __init__(self):
        self.attn_map = "stale"
        self.received = None

    def set_attn_map(self, value):
        self.attn_map = value

    def step(self, *args, **kwargs):
        self.received = kwargs
        return ("next",)


class PipelineSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = {name: (ROOT / name).read_text(encoding="utf-8")
                       for name in (*PIPELINES, SCHEDULER)}
        cls.trees = {name: ast.parse(source, filename=name)
                     for name, source in cls.sources.items()}
        cls.methods = {}
        for name in PIPELINES:
            pipeline = next(node for node in cls.trees[name].body
                            if isinstance(node, ast.ClassDef)
                            and node.name == "FluxKontextEditPipelineAMO")
            cls.methods[name] = {node.name: node for node in pipeline.body
                                 if isinstance(node, ast.FunctionDef)}

    def test_sources_compile_and_unused_strength_is_removed(self):
        for name, tree in self.trees.items():
            with self.subTest(file=name):
                compile(tree, name, "exec")
                if name in PIPELINES:
                    self.assertNotIn("edit_strength", self.sources[name])
                    helper = self.methods[name]["_compute_velocity_delta"]
                    self.assertIn("generator", [arg.arg for arg in helper.args.args])

    def test_noise_sampling_forwards_single_and_list_generators(self):
        for name, tree in self.trees.items():
            self.assertFalse(calls(tree, "randn") + calls(tree, "randn_like"), name)
            sampling = calls(tree, "randn_tensor")
            self.assertTrue(sampling, name)
            for call in sampling:
                for generator in (object(), [object(), object()]):
                    with self.subTest(file=name, line=call.lineno, batch=isinstance(generator, list)):
                        seen = []
                        env = dict(
                            generator=generator, shape=(2, 3, 4),
                            src_latents=SimpleNamespace(shape=(2, 3, 4)),
                            sample=SimpleNamespace(shape=(2, 3, 4), device="cpu", dtype="float32"),
                            device="cpu", dtype="float32",
                            randn_tensor=lambda *args, **kwargs: seen.append((args, kwargs)),
                        )
                        eval(compile(ast.Expression(call), "<sampling>", "eval"), env)
                        self.assertIs(seen[0][1]["generator"], generator)
                        self.assertEqual(seen[0][0][0], (2, 3, 4))
                        self.assertEqual(seen[0][1]["device"], "cpu")
                        self.assertEqual(seen[0][1]["dtype"], "float32")

    def test_scheduler_step_and_helper_forward_generator(self):
        for name in PIPELINES:
            method = self.methods[name]["__call__"]
            steps = calls(method, "step")
            self.assertEqual(len(steps), 1, name)
            for generator in (object(), [object(), object()]):
                with self.subTest(file=name, batch=isinstance(generator, list)):
                    scheduler = SchedulerStub()
                    eval(compile(ast.Expression(steps[0]), "<step>", "eval"), dict(
                        self=SimpleNamespace(scheduler=scheduler),
                        noise_pred=1, t=1, latents=1, generator=generator,
                    ))
                    self.assertIs(scheduler.received["generator"], generator)
            for call in calls(method, "_compute_velocity_delta"):
                kwargs = {item.arg: item.value for item in call.keywords}
                self.assertIsInstance(kwargs.get("generator"), ast.Name)
                self.assertEqual(kwargs["generator"].id, "generator")

    def test_initial_attention_reset_and_unsupported_scheduler(self):
        for name in PIPELINES:
            with self.subTest(file=name):
                body = self.methods[name]["__call__"].body
                start = next(index for index, node in enumerate(body)
                             if isinstance(node, ast.Expr) and calls(node, "init_forwards"))
                guards = body[start + 1:start + 3]
                self.assertTrue(all(isinstance(node, ast.If) for node in guards))
                for enabled in (False, True):
                    scheduler = SchedulerStub()
                    execute(guards, self=SimpleNamespace(scheduler=scheduler), use_att=enabled)
                    self.assertIsNone(scheduler.attn_map)
                execute(guards, self=SimpleNamespace(scheduler=object()), use_att=False)
                with self.assertRaisesRegex(ValueError, "set_attn_map"):
                    execute(guards, self=SimpleNamespace(scheduler=object()), use_att=True)

    def test_disabled_attention_clears_state_only_when_supported(self):
        for name in PIPELINES:
            with self.subTest(file=name):
                blocks = [node for node in ast.walk(self.methods[name]["__call__"])
                          if isinstance(node, ast.If)
                          and any(isinstance(call.args[0], ast.Name)
                                  and call.args[0].id == "attn_map_norm"
                                  for statement in node.body
                                  for call in calls(statement, "set_attn_map") if call.args)]
                self.assertEqual(len(blocks), 1)
                scheduler = SchedulerStub()
                execute(blocks[0].orelse, self=SimpleNamespace(scheduler=scheduler))
                self.assertIsNone(scheduler.attn_map)
                execute(blocks[0].orelse, self=SimpleNamespace(scheduler=object()))

    def test_effective_trajectory_updates_remain(self):
        for name in PIPELINES:
            with self.subTest(file=name):
                source = self.sources[name]
                self.assertIn("zt_edit = zt_edit + (t_im1 - t_i) * V_delta", source)
                self.assertIn("guidance_scale * 0.6", source)
                if name == "pipeline_flux_kontext_stepwise_injection.py":
                    self.assertIn("latents = zt_edit.clone()", source)
                    self.assertIn("zt_edit = latents.clone()", source)


class TransformerOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (ROOT / "outputs.py").read_text(encoding="utf-8")
        output = next(node for node in ast.parse(source).body
                      if isinstance(node, ast.ClassDef)
                      and node.name == "Transformer2DModelOutput")
        # Execute only the output class, replacing diffusers.BaseOutput with a stub.
        env = {"dataclass": dataclass, "BaseOutput": object, "__name__": __name__}
        exec(compile(ast.Module(body=[output], type_ignores=[]), "<output>", "exec"), env)
        cls.output_type = env["Transformer2DModelOutput"]

    def test_output_without_attention(self):
        sample = object()
        output = self.output_type(sample=sample)
        self.assertIs(output.sample, sample)
        self.assertIsNone(output.attn_map)
        with self.assertRaises(TypeError):
            self.output_type()

    def test_output_with_attention(self):
        sample, attention = object(), [object()]
        output = self.output_type(sample=sample, attn_map=attention)
        self.assertIs(output.sample, sample)
        self.assertIs(output.attn_map, attention)


if __name__ == "__main__":
    unittest.main()
