"""Export MBHT/P3-compatible four-prefix states from RecBole teachers.

Supported teachers are the RecBole 1.2.1 implementations of SASRec, GRU4Rec,
FEARec and CORE. Every prefix is run independently through the teacher, so the saved
states are causal.  The exporter reads fixed benchmark splits without
shuffling and never trains a missing teacher implicitly.
"""

import argparse
import hashlib
from logging import getLogger
import os
from pathlib import Path

import torch

from recbole.config import Config
from recbole.data import create_dataset
from recbole.data.utils import get_dataloader
from recbole.utils import get_model, init_logger, init_seed


SUPPORTED_TEACHERS = (
    "SASRec",
    "GRU4Rec",
    "FEARec",
    "CORE",
    "LightSANs",
)
BENCHMARK_PHASES = ("train", "valid", "test")
PREFIX_RATIOS = (0.25, 0.50, 0.75, 1.00)


def get_args():
    parser = argparse.ArgumentParser(
        description="Export four causal prefix states from a RecBole teacher."
    )
    parser.add_argument(
        "--teacher_model", choices=SUPPORTED_TEACHERS, required=True
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument(
        "--split",
        nargs="+",
        choices=BENCHMARK_PHASES,
        default=["train"],
        help="One or more fixed benchmark splits; default: train.",
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--topk", type=int, default=100,
                        help="Accepted for four-export CLI compatibility.")
    parser.add_argument("--item_chunk_size", type=int, default=10000,
                        help="Accepted for four-export CLI compatibility.")
    parser.add_argument("--output_dir", default="teacher_trajectories")
    parser.add_argument(
        "--save_dtype", choices=("float16", "float32"), default="float16"
    )
    return parser.parse_args()


def load_torch_file(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def item_mapping_sha256(dataset, item_field):
    try:
        tokens = dataset.field2id_token[item_field]
    except (AttributeError, KeyError) as error:
        raise ValueError(
            f"Cannot read token-to-ID mapping for {item_field!r}."
        ) from error
    serialized = "\n".join(map(str, tokens)).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def get_checkpoint_value(checkpoint, key):
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is None:
        return None
    try:
        return checkpoint_config[key]
    except (KeyError, TypeError):
        return None


def checkpoint_matches(checkpoint, model_name, dataset_name, seed):
    checkpoint_model = get_checkpoint_value(checkpoint, "model")
    checkpoint_dataset = get_checkpoint_value(checkpoint, "dataset")
    checkpoint_seed = get_checkpoint_value(checkpoint, "seed")
    if checkpoint_model is None or checkpoint_dataset is None:
        return False
    try:
        checkpoint_seed = int(checkpoint_seed)
    except (TypeError, ValueError):
        return False
    return (
        str(checkpoint_model).lower() == model_name.lower()
        and str(checkpoint_dataset) == dataset_name
        and checkpoint_seed == int(seed)
    )


def find_unique_checkpoint(config, model_name, dataset_name, seed):
    checkpoint_dir = Path(config["checkpoint_dir"])
    candidates = sorted(checkpoint_dir.glob(f"{model_name}-*.pth"))
    matches = []
    unreadable = []
    for candidate in candidates:
        try:
            checkpoint = load_torch_file(candidate)
        except Exception as error:
            unreadable.append(f"{candidate}: {error}")
            continue
        if isinstance(checkpoint, dict) and checkpoint_matches(
            checkpoint, model_name, dataset_name, seed
        ):
            matches.append(candidate)
    if len(matches) > 1:
        listing = "\n".join(f"  - {path}" for path in matches)
        raise RuntimeError(
            "Multiple matching checkpoints were found. Pass --checkpoint "
            f"explicitly:\n{listing}"
        )
    if len(matches) == 1:
        return matches[0]
    details = (
        f"No checkpoint matches model={model_name}, dataset={dataset_name}, "
        f"seed={seed} in {checkpoint_dir}."
    )
    if candidates:
        details += " Candidate files:\n" + "\n".join(
            f"  - {path}" for path in candidates
        )
    if unreadable:
        details += "\nUnreadable candidates:\n" + "\n".join(
            f"  - {entry}" for entry in unreadable
        )
    raise FileNotFoundError(details)


def validate_benchmark_files(config):
    if list(config["benchmark_filename"] or []) != list(BENCHMARK_PHASES):
        raise ValueError(
            "The config must contain benchmark_filename: [train, valid, test]."
        )
    dataset_name = config["dataset"]
    dataset_path = Path(config["data_path"])
    user_field = config["USER_ID_FIELD"]
    item_field = config["ITEM_ID_FIELD"]
    item_list_field = item_field + config["LIST_SUFFIX"]
    required = {
        f"{user_field}:token",
        f"{item_list_field}:token_seq",
        f"{item_field}:token",
    }
    for phase in BENCHMARK_PHASES:
        path = dataset_path / f"{dataset_name}.{phase}.inter"
        if not path.is_file():
            raise FileNotFoundError(f"Missing fixed benchmark file: {path}")
        with path.open("r", encoding="utf-8") as file:
            header = file.readline().rstrip("\r\n").split("\t")
        missing = required.difference(header)
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing)}"
            )


def build_config(args):
    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config does not exist: {config_path}")
    overrides = {"gpu_id": args.gpu_id, "seed": args.seed}
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        overrides["train_batch_size"] = args.batch_size
        overrides["eval_batch_size"] = args.batch_size
    return Config(
        model=args.teacher_model,
        dataset=args.dataset,
        config_file_list=[str(config_path)],
        config_dict=overrides,
    )


def checkpoint_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("RecBole checkpoint must contain a dictionary.")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise KeyError("RecBole checkpoint is missing state_dict.")
    return state_dict


def validate_checkpoint(checkpoint, config, checkpoint_path):
    checkpoint_model = get_checkpoint_value(checkpoint, "model")
    checkpoint_dataset = get_checkpoint_value(checkpoint, "dataset")
    checkpoint_seed = get_checkpoint_value(checkpoint, "seed")
    if checkpoint_model is None or str(checkpoint_model).lower() != str(
        config["model"]
    ).lower():
        raise ValueError(
            f"Checkpoint model mismatch: requested {config['model']}, "
            f"checkpoint contains {checkpoint_model} ({checkpoint_path})."
        )
    if str(checkpoint_dataset) != str(config["dataset"]):
        raise ValueError(
            f"Checkpoint dataset mismatch: requested {config['dataset']}, "
            f"checkpoint contains {checkpoint_dataset} ({checkpoint_path})."
        )
    try:
        checkpoint_seed = int(checkpoint_seed)
    except (TypeError, ValueError) as error:
        raise ValueError("Checkpoint does not contain a valid seed.") from error
    if checkpoint_seed != int(config["seed"]):
        raise ValueError(
            f"Checkpoint seed={checkpoint_seed}, requested seed={config['seed']}."
        )


def load_model(config, train_dataset, checkpoint_path):
    checkpoint = load_torch_file(checkpoint_path)
    validate_checkpoint(checkpoint, config, checkpoint_path)
    model = get_model(config["model"])(config, train_dataset)
    model.load_state_dict(checkpoint_state_dict(checkpoint), strict=True)
    other_parameter = checkpoint.get("other_parameter")
    if other_parameter is not None and hasattr(model, "load_other_parameter"):
        model.load_other_parameter(other_parameter)
    model = model.to(config["device"])
    model.eval()
    return model


def get_item_embedding(model, item_num):
    embedding_layer = getattr(model, "item_embedding", None)
    if embedding_layer is None or not hasattr(embedding_layer, "weight"):
        raise AttributeError("Teacher model has no item_embedding.weight.")
    weight = embedding_layer.weight
    if weight.dim() != 2 or weight.size(0) != int(item_num):
        raise ValueError(
            "item_embedding.weight must have exact shape [item_num, D]; "
            f"got {tuple(weight.shape)}, item_num={item_num}."
        )
    return weight


def is_core_teacher(model):
    return model.__class__.__name__.lower() == "core"


def encode_teacher_state(model, item_seq, item_seq_len):
    if is_core_teacher(model):
        return model.forward(item_seq)
    return model.forward(item_seq, item_seq_len)


def teacher_score_rule(model):
    if is_core_teacher(model):
        return "l2_normalized_state_dot_l2_normalized_item_div_temperature"
    return "state_dot_item_embedding"


def teacher_score_temperature(model):
    if not is_core_teacher(model):
        return 1.0
    temperature = float(getattr(model, "temperature", 0.0))
    if temperature <= 0:
        raise ValueError(
            f"CORE temperature must be positive, got {temperature}."
        )
    return temperature


def four_prefix_lengths(sequence_lengths):
    if sequence_lengths.dim() != 1:
        raise ValueError("sequence_lengths must have shape [B].")
    if torch.any(sequence_lengths <= 0):
        raise ValueError("Every sequence must contain at least one item.")
    lengths = sequence_lengths.long()
    prefix_lengths = torch.stack(
        (
            (lengths + 3) // 4,
            (lengths + 1) // 2,
            (3 * lengths + 3) // 4,
            lengths,
        ),
        dim=1,
    )
    prefix_mask = torch.ones_like(prefix_lengths, dtype=torch.bool)
    prefix_mask[:, 1:] = prefix_lengths[:, 1:] != prefix_lengths[:, :-1]
    return prefix_lengths, prefix_mask


def truncate_batch(item_seq, prefix_lengths):
    positions = torch.arange(
        item_seq.size(1), device=item_seq.device
    ).unsqueeze(0)
    return item_seq.masked_fill(positions >= prefix_lengths.unsqueeze(1), 0)


def make_export_loader(config, split_dataset):
    return get_dataloader(config, "train")(
        config, split_dataset, None, shuffle=False
    )


def assert_unique_composite_keys(sample_ids, sequence_lengths, targets):
    seen = set()
    duplicates = []
    for key in zip(
        sample_ids.tolist(), sequence_lengths.tolist(), targets.tolist()
    ):
        if key in seen:
            duplicates.append(key)
            if len(duplicates) == 10:
                break
        seen.add(key)
    if duplicates:
        raise ValueError(
            "Duplicate (sample_id, sequence_length, target_item) keys: "
            f"{duplicates}"
        )


def export_split(
    model,
    split_dataset,
    config,
    phase,
    checkpoint_path,
    checkpoint_digest,
    output_path,
    save_dtype,
):
    model.eval()
    device = next(model.parameters()).device
    user_field = config["USER_ID_FIELD"]
    item_field = config["ITEM_ID_FIELD"]
    item_list_field = item_field + config["LIST_SUFFIX"]
    length_field = config["ITEM_LIST_LENGTH_FIELD"]
    item_num = int(split_dataset.item_num)
    item_embedding = get_item_embedding(model, item_num)
    hidden_size = int(item_embedding.size(1))
    stored_dtype = torch.float16 if save_dtype == "float16" else torch.float32

    sample_count = len(split_dataset)
    sample_ids_store = torch.empty(sample_count, dtype=torch.long)
    targets_store = torch.empty(sample_count, dtype=torch.long)
    sequence_lengths_store = torch.empty(sample_count, dtype=torch.long)
    prefix_lengths_store = torch.empty((sample_count, 4), dtype=torch.long)
    prefix_mask_store = torch.empty((sample_count, 4), dtype=torch.bool)
    trajectory_states_store = torch.empty(
        (sample_count, 4, hidden_size), dtype=stored_dtype
    )

    cursor = 0
    loader = make_export_loader(config, split_dataset)
    with torch.no_grad():
        for interaction in loader:
            required = {user_field, item_field, item_list_field}
            missing = required.difference(interaction.interaction.keys())
            if missing:
                raise KeyError(
                    f"Export batch is missing fields: {sorted(missing)}"
                )
            sample_ids = interaction[user_field].detach().cpu().long()
            targets = interaction[item_field].detach().cpu().long()
            if length_field in interaction.interaction:
                sequence_lengths = interaction[length_field].detach().cpu().long()
            else:
                sequence_lengths = (
                    interaction[item_list_field] != 0
                ).sum(dim=1).cpu().long()

            interaction = interaction.to(device)
            item_seq = interaction[item_list_field]
            actual_lengths = (item_seq != 0).sum(dim=1).long()
            if not torch.equal(actual_lengths.cpu(), sequence_lengths):
                raise ValueError(
                    "Stored sequence lengths do not match non-padding items."
                )
            prefix_lengths, prefix_mask = four_prefix_lengths(actual_lengths)

            prefix_states = []
            for prefix_index in range(4):
                current_lengths = prefix_lengths[:, prefix_index]
                prefix_items = truncate_batch(item_seq, current_lengths)
                state = encode_teacher_state(
                    model, prefix_items, current_lengths
                )
                if state.dim() != 2 or state.size(1) != hidden_size:
                    raise ValueError(
                        "Teacher forward must return [B, embedding_dim]; got "
                        f"{tuple(state.shape)}, expected dim={hidden_size}."
                    )
                prefix_states.append(state)
            prefix_states = torch.stack(prefix_states, dim=1)
            if not torch.isfinite(prefix_states).all():
                raise RuntimeError("Teacher trajectory contains NaN or Inf.")

            batch_size = sample_ids.numel()
            end = cursor + batch_size
            if end > sample_count:
                raise RuntimeError("Exporter produced more rows than the split.")
            sample_ids_store[cursor:end] = sample_ids
            targets_store[cursor:end] = targets
            sequence_lengths_store[cursor:end] = sequence_lengths
            prefix_lengths_store[cursor:end] = prefix_lengths.cpu()
            prefix_mask_store[cursor:end] = prefix_mask.cpu()
            stored_states = prefix_states.to(device="cpu", dtype=stored_dtype)
            if not torch.isfinite(stored_states).all():
                raise RuntimeError(
                    "Teacher states overflowed in the selected storage dtype."
                )
            trajectory_states_store[cursor:end] = stored_states
            cursor = end

    if cursor != sample_count:
        raise RuntimeError(
            f"Exporter wrote {cursor} rows, expected {sample_count}."
        )
    if torch.any(targets_store <= 0) or torch.any(targets_store >= item_num):
        raise ValueError("target_items contains padding or out-of-range IDs.")
    assert_unique_composite_keys(
        sample_ids_store, sequence_lengths_store, targets_store
    )

    payload = {
        "sample_ids": sample_ids_store,
        "target_items": targets_store,
        "sequence_lengths": sequence_lengths_store,
        "prefix_lengths": prefix_lengths_store,
        "prefix_mask": prefix_mask_store,
        "trajectory_states": trajectory_states_store,
        "metadata": {
            "format_version": 1,
            "teacher": str(config["model"]),
            "dataset": str(config["dataset"]),
            "phase": phase,
            "seed": int(config["seed"]),
            "prefix_ratios": list(PREFIX_RATIOS),
            "num_prefixes": 4,
            "hidden_size": hidden_size,
            "state_dtype": save_dtype,
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "checkpoint_sha256": checkpoint_digest,
            "item_num": item_num,
            "item_mapping_sha256": item_mapping_sha256(
                split_dataset, item_field
            ),
            "padding_item_id": 0,
            "causal_prefix_export": True,
            "prefix_definition": "ceil(L/4),ceil(L/2),ceil(3L/4),L",
            "short_sequence_duplicates_masked": True,
            "shuffle": False,
            "teacher_readout": (
                f"RecBole {config['model']} forward output after separately "
                "truncated prefix"
            ),
            "teacher_score_rule": teacher_score_rule(model),
            "score_temperature": teacher_score_temperature(model),
            "trajectory_states_l2_normalized": is_core_teacher(model),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(output_path))
    print(f"Saved {config['model']} four-prefix trajectory: {output_path}")
    print(f"sample_ids shape: {tuple(sample_ids_store.shape)}")
    print(f"prefix_lengths shape: {tuple(prefix_lengths_store.shape)}")
    print(f"trajectory_states shape: {tuple(trajectory_states_store.shape)}")
    print(f"trajectory dtype: {trajectory_states_store.dtype}")


def main():
    args = get_args()
    if args.topk <= 0:
        raise ValueError("topk must be positive.")
    if args.item_chunk_size <= 0:
        raise ValueError("item_chunk_size must be positive.")
    config = build_config(args)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info("PID: %s", os.getpid())
    logger.info("Exporter arguments: %s", args)
    validate_benchmark_files(config)

    dataset = create_dataset(config)
    split_datasets = dataset.build()
    if len(split_datasets) != 3:
        raise RuntimeError(
            f"Expected three benchmark splits, got {len(split_datasets)}."
        )
    if list(config["benchmark_filename"]) != list(BENCHMARK_PHASES):
        raise RuntimeError(
            f"Unexpected benchmark order: {config['benchmark_filename']}."
        )

    if args.checkpoint is None:
        checkpoint_path = find_unique_checkpoint(
            config, args.teacher_model, args.dataset, args.seed
        )
    else:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Explicit checkpoint does not exist: {checkpoint_path}"
            )

    model = load_model(config, split_datasets[0], checkpoint_path)
    checkpoint_digest = file_sha256(checkpoint_path)
    output_dir = Path(args.output_dir)
    for phase in args.split:
        split_index = BENCHMARK_PHASES.index(phase)
        output_path = output_dir / (
            f"{args.teacher_model}_trajectory4_{args.dataset}_"
            f"{phase}_seed{args.seed}.pt"
        )
        export_split(
            model=model,
            split_dataset=split_datasets[split_index],
            config=config,
            phase=phase,
            checkpoint_path=checkpoint_path,
            checkpoint_digest=checkpoint_digest,
            output_path=output_path,
            save_dtype=args.save_dtype,
        )


if __name__ == "__main__":
    main()
