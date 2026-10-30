"""CPU-only regression tests for CLI, resume identity and distributed control flow."""

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import math
from pathlib import Path
import sys
import struct
import zlib
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import inference_utils as safety

ENTRIES = [
    "infer_flux_text.py",
    "infer_flux_text_multi_gpu.py",
    "infer_flux_color_multi_gpu.py",
    "infer_longcat_attention.py",
    "infer_longcat_text.py",
    "infer_longcat_text_multi_gpu.py",
]
MULTI = [ENTRIES[1], ENTRIES[2], ENTRIES[5]]


def tree(name):
    return ast.parse((ROOT / name).read_text(encoding="utf-8"), filename=name)


def cli(name, argv=()):
    main = next(node for node in tree(name).body if isinstance(node, ast.If))
    captured = []
    namespace = {"argparse": argparse, "run": captured.append, "validate_args": safety.validate_args}
    with patch.object(sys, "argv", [name, *argv]):
        with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            exec(compile(ast.Module(body=main.body, type_ignores=[]), name, "exec"), namespace)
    return captured[0]


class InferenceSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = [{"image_path": f"group{i}/same.png", "text_s": "OPEN", "text_t": "CLOSED"} for i in range(3)]
        self.options = {"json_path": "unused.json", "exp_dir": "unused", "seed": 12}

    def test_sample_names_and_seeds_are_stable_and_distinct(self):
        names = [safety.output_path("out", item) for item in self.data]
        self.assertEqual(len(set(names)), 3)
        changed = dict(self.data[0], text_t="OTHER")
        self.assertNotEqual(safety.sample_id(changed), safety.sample_id(self.data[0]))
        seeds = {safety.sample_id(item): safety.sample_seed(12, item) for item in self.data}
        for item in self.data[1:][::-1]:
            self.assertEqual(safety.sample_seed(12, item), seeds[safety.sample_id(item)])
        self.assertNotEqual(safety.sample_seed(13, self.data[0]), safety.sample_seed(12, self.data[0]))
        self.assertEqual(safety.sample_id(dict(self.data[0], id="stable")), safety.sample_id(dict(changed, id="stable")))
        self.assertNotEqual(safety.sample_id(dict(changed, id=1)), safety.sample_id(dict(changed, id="1")))
        self.assertEqual(Path(safety.output_path(self.root, dict(changed, image_path="../../unsafe?.png"))).parent, self.root)

    def test_manifest_rejects_changes_and_legacy_outputs(self):
        self.assertEqual(safety.prepare_run(self.root, self.options, ENTRIES[0], self.data), self.data)
        safety.prepare_run(self.root, dict(self.options, exp_dir="elsewhere", json_path="moved.json"), ENTRIES[0], self.data)
        for options, data, entry in [
            (dict(self.options, seed=13), self.data, ENTRIES[0]),
            (self.options, self.data[:-1], ENTRIES[0]),
            (self.options, self.data, ENTRIES[1]),
        ]:
            with self.subTest(options=options, entry=entry), self.assertRaises(ValueError):
                safety.prepare_run(self.root, options, entry, data)
        legacy = self.root / "legacy"
        legacy.mkdir()
        Image.new("RGB", (2, 2)).save(legacy / "old.png")
        with self.assertRaisesRegex(ValueError, "no run_config"):
            safety.prepare_run(legacy, self.options, ENTRIES[0], self.data)

    def test_manifest_concurrent_creation_and_validation(self):
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda _: safety.prepare_run(self.root, self.options, ENTRIES[0], self.data), range(24)))
        self.assertTrue(all(result == self.data for result in results))
        self.assertEqual(json.loads((self.root / "run_config.json").read_text(encoding="utf-8"))["schema_version"], 1)
        self.assertFalse(list(self.root.glob(".config-*")))
        for data in [self.data + [self.data[0]], [{"image_path": "x"}], {}, [dict(self.data[0], id=True)]]:
            with self.subTest(data=data), self.assertRaises(ValueError):
                safety.prepare_run(self.root, self.options, ENTRIES[0], data)

    def test_atomic_image_save_and_corrupt_output_repair(self):
        path = self.root / "result.png"
        path.write_bytes(b"unfinished")
        self.assertFalse(safety.is_output_complete(path))
        safety.atomic_save_image(Image.new("RGB", (3, 4)), path)
        self.assertTrue(safety.is_output_complete(path))
        original = path.read_bytes()
        class BrokenImage:
            def save(self, temporary, format):
                Path(temporary).write_bytes(b"partial")
                raise OSError("interrupted")
        with self.assertRaises(OSError):
            safety.atomic_save_image(BrokenImage(), path)
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse(list(self.root.glob(".image-*")))

    def test_corrupt_compressed_pixels_not_skipped(self):
        def chunk(kind, payload):
            return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        path = self.root / "invalid-pixels.png"
        for size in [(1, 1), (100000, 100000)]:
            with self.subTest(dimensions=size):
                path.write_bytes(
                    bytes.fromhex("89504e470d0a1a0a")
                    + chunk(b"IHDR", struct.pack(">2I5B", *size, 8, 2, 0, 0, 0))
                    + chunk(b"IDAT", b"invalid-zlib")
                    + chunk(b"IEND", b"")
                )
                if size == (1, 1):
                    with Image.open(path) as image:
                        image.verify()  # CRCs alone do not validate compressed pixels.
                self.assertFalse(safety.is_output_complete(path))

    def test_all_cli_attention_switches_and_removed_option(self):
        for name in ENTRIES:
            with self.subTest(entry=name):
                self.assertEqual(cli(name).use_att, name != ENTRIES[5])
                self.assertTrue(cli(name, ["--use_att"]).use_att)
                self.assertFalse(cli(name, ["--no-use_att"]).use_att)
                self.assertFalse(cli(name, ["--scheduler", "euler", "--no-use_att"]).use_att)
                for argv in [["--use_att", "False"], ["--edit_strength", "0.5"], ["--scheduler", "euler", "--use_att"], ["--num_inference_steps", "0"]]:
                    with self.assertRaises(SystemExit):
                        cli(name, argv)
                if name != ENTRIES[3]:
                    for argv in [["--use_flowedit", "--n_avg", "0"], ["--use_flowedit", "--flowedit_start_step", "50"]]:
                        with self.assertRaises(SystemExit):
                            cli(name, argv)


class Progress(list):
    def __init__(self, data, **kwargs):
        super().__init__(data)

    @staticmethod
    def write(message):
        print(message)

    def set_postfix(self, *args, **kwargs):
        pass


class EntryControlFlowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def execute(self, name, data, rank=0, world=1, auto=False, fail=False):
        dataset = self.root / "dataset.json"
        dataset.write_text(json.dumps(data), encoding="utf-8")
        args = cli(ENTRIES[0] if name == "infer_flux_baseline.py" else name)
        args.json_path = str(dataset)
        args.exp_dir = str(self.root / name)
        args.use_flowedit = name != ENTRIES[3]
        args.auto_src_prompt = auto
        state = SimpleNamespace(loads=0, calls=[], barriers=0, reductions=0)
        class Generator:
            def __init__(self, device):
                self.device = device
            def manual_seed(self, seed):
                self.seed = seed
                return self
        class Scheduler:
            config = {}
            @classmethod
            def from_config(cls, config):
                return cls()
            def set_c(self, c):
                pass
            def set_overshot_func(self, function):
                pass
        class Pipeline:
            scheduler = Scheduler()
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                state.loads += 1
                return cls()
            def to(self, device):
                state.device = device
                return self
            def __call__(self, **kwargs):
                state.calls.append(kwargs)
                if fail:
                    raise RuntimeError("simulated generation failure")
                return SimpleNamespace(images=[Image.new("RGB", (2, 2))])
        class Accelerator:
            process_index = rank
            num_processes = world
            device = f"local-device-{rank}"
            is_main_process = rank == 0
            is_local_main_process = rank == 0
            def __init__(self, **kwargs):
                pass
            def wait_for_everyone(self):
                state.barriers += 1
            def reduce(self, value, reduction):
                state.reductions += 1
                self_reduction = reduction
                if self_reduction != "sum":
                    raise AssertionError(reduction)
                return value
        namespace = dict(vars(safety))
        namespace.update({
            "__file__": str(ROOT / name), "os": __import__("os"), "math": math,
            "time": SimpleNamespace(time=lambda: 0), "Image": Image,
            "tqdm": Progress,
            "torch": SimpleNamespace(bfloat16="bf16", Generator=Generator, tensor=lambda value, **kwargs: SimpleNamespace(item=lambda: value)),
            "Accelerator": Accelerator, "InitProcessGroupKwargs": lambda **kwargs: kwargs,
            "timedelta": __import__("datetime").timedelta,
            "FluxKontextEditPipelineAMO": Pipeline, "LongCatImageEditAMOPipeline": Pipeline,
            "FluxKontextPipeline": Pipeline, "json_path": str(dataset), "file_save_dir": args.exp_dir,
            "StochasticRFOvershotDiscreteScheduler": Scheduler,
            "load_image": lambda _: Image.new("RGB", (2, 2)),
        })
        node = next(node for node in tree(name).body if isinstance(node, ast.FunctionDef) and node.name == "run")
        exec(compile(ast.Module(body=[node], type_ignores=[]), name, "exec"), namespace)
        output = io.StringIO()
        try:
            with redirect_stdout(output):
                namespace["run"]() if name == "infer_flux_baseline.py" else namespace["run"](args)
        except Exception as error:
            raise RuntimeError(f"{error}\n{output.getvalue()}") from error
        return state, args

    def records(self, count):
        records = []
        for index in range(count):
            path = self.root / f"input{index}.png"
            Image.new("RGB", (2, 2)).save(path)
            records.append({"image_path": str(path), "text_s": "OPEN", "text_t": "CLOSED"})
        return records

    def test_single_entries_smoke_and_stable_resume(self):
        data = self.records(3)
        for name in [ENTRIES[0], ENTRIES[3], ENTRIES[4], "infer_flux_baseline.py"]:
            with self.subTest(entry=name):
                state, args = self.execute(name, data)
                self.assertEqual(len(state.calls), len(data))
                for call, item in zip(state.calls, data):
                    self.assertEqual(call["generator"].seed, safety.sample_seed(args.seed, item))
                    self.assertNotIn("edit_strength", call)
                resumed, _ = self.execute(name, data)
                self.assertFalse(resumed.calls)

    def test_multi_resume_after_changing_world_size(self):
        data = self.records(3)
        for name in MULTI:
            with self.subTest(entry=name):
                first, _ = self.execute(name, data, rank=0, world=2)
                self.assertEqual(len(first.calls), 2)
                resumed = [self.execute(name, data, rank=rank, world=4)[0] for rank in range(4)]
                self.assertEqual([state.loads for state in resumed], [0, 0, 1, 0])
                self.assertEqual(resumed[2].calls[0]["generator"].seed, safety.sample_seed(cli(name).seed, data[2]))

    def test_longcat_source_manual_auto_and_seed(self):
        data = self.records(1)
        for index, auto in enumerate([False, True]):
            for name in [ENTRIES[4], ENTRIES[5]]:
                with self.subTest(entry=name, automatic=auto):
                    data[0]["id"] = f"case-{index}"
                    # Different configuration requires its own output directory.
                    with tempfile.TemporaryDirectory(dir=self.root) as separate:
                        previous = self.root
                        self.root = Path(separate)
                        try:
                            state, args = self.execute(name, data, auto=auto)
                        finally:
                            self.root = previous
                    self.assertEqual(len(state.calls), 1)
                    call = state.calls[0]
                    self.assertEqual(call["src_prompt"], None if auto else "A photo with the word 'OPEN'")
                    self.assertEqual(call["generator"].seed, safety.sample_seed(args.seed, data[0]))

    def test_multi_empty_ranks_and_fully_resumed_runs(self):
        for name in MULTI:
            with self.subTest(entry=name):
                data = self.records(1)
                states = [self.execute(name, data, rank=rank, world=4)[0] for rank in range(4)]
                self.assertEqual([state.loads for state in states], [1, 0, 0, 0])
                self.assertTrue(all(state.barriers == 2 and state.reductions == 1 for state in states))
                self.assertEqual(states[0].device, "local-device-0")
                self.assertEqual(states[0].calls[0]["generator"].device, "local-device-0")
                resumed = [self.execute(name, data, rank=rank, world=2)[0] for rank in range(2)]
                self.assertTrue(all(state.loads == 0 and state.barriers == 2 and state.reductions == 1 for state in resumed))

    def test_multi_empty_dataset_and_visible_failures(self):
        for name in MULTI:
            with self.subTest(entry=name):
                state, _ = self.execute(name, [], rank=3, world=4)
                self.assertEqual((state.loads, state.barriers, state.reductions), (0, 2, 1))
                # Keep the prior empty dataset manifest isolated.
                with tempfile.TemporaryDirectory(dir=self.root) as separate:
                    previous = self.root
                    self.root = Path(separate)
                    try:
                        with self.assertRaisesRegex(RuntimeError, "Inference failed"):
                            self.execute(name, self.records(1), fail=True)
                    finally:
                        self.root = previous


if __name__ == "__main__":
    unittest.main()
