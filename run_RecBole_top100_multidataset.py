"""Export MBHT/P3-compatible Top-K artifacts from RecBole teachers.

Supported teachers: SASRec, GRU4Rec, FEARec and CORE from RecBole 1.2.1.

The exporter reads pre-split benchmark files, preserves their sample IDs and
row order, masks padding item 0 and items already present in each sample's
history, and computes Top-K scores in item chunks.  It never silently trains a
teacher.  Automatic training is available only through
``--train_if_missing true``.
"""

import argparse
import hashlib
from logging import getLogger
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.utils import get_dataloader
from recbole.utils import (
    get_model,
    get_trainer,
    init_logger,
    init_seed,
)


SUPPORTED_TEACHERS = (
    "SASRec",
    "GRU4Rec",
    "FEARec",
    "CORE",
    "LightSANs",
)
BENCHMARK_PHASES = ("train", "valid", "test")


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Expected a boolean value, got {value!r}."
    )


def get_args():
    parser = argparse.ArgumentParser(
        description="Export RecBole teacher Top-K logits for P3 distillation."
    )
    parser.add_argument(
        "--teacher_model",
        choices=SUPPORTED_TEACHERS,
        required=True,
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
        default=list(BENCHMARK_PHASES),
        help="One or more fixed benchmark splits to export.",
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--output_dir", default="teacher_logits")
    parser.add_argument("--item_chunk_size", type=int, default=10000)
    parser.add_argument(
        "--train_if_missing",
        type=parse_bool,
        default=False,
        help=(
            "Train only when no matching checkpoint exists. Multiple matching "
            "checkpoints always cause an error."
        ),
    )
    return parser.parse_args()


def load_torch_file(path):
    """Load trusted local RecBole checkpoints across PyTorch versions."""
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
        except Exception as error:  # report all unusable candidates together
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

    details = [f"No checkpoint matches model={model_name}, "]
    details.append(f"dataset={dataset_name}, seed={seed} in {checkpoint_dir}.")
    if candidates:
        details.append(" Candidate files:\n")
        details.append("\n".join(f"  - {path}" for path in candidates))
    if unreadable:
        details.append("\n Unreadable candidates:\n")
        details.append("\n".join(f"  - {entry}" for entry in unreadable))
    raise FileNotFoundError("".join(details))


def validate_benchmark_files(config):
    benchmark_names = config["benchmark_filename"]
    if list(benchmark_names or []) != list(BENCHMARK_PHASES):
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
    overrides = {
        "gpu_id": args.gpu_id,
        "seed": args.seed,
    }
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


def train_teacher(config, dataset, logger):
    logger.info("No checkpoint found; --train_if_missing=true, starting training.")
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = get_model(config["model"])(config, train_data.dataset).to(
        config["device"]
    )
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(
        config, model
    )
    trainer.fit(
        train_data,
        valid_data,
        saved=True,
        show_progress=config["show_progress"],
    )
    checkpoint_path = Path(trainer.saved_model_file)
    if not checkpoint_path.is_file():
        raise RuntimeError(
            f"Training finished but checkpoint was not saved: {checkpoint_path}"
        )
    split_datasets = [
        train_data.dataset,
        valid_data.dataset,
        test_data.dataset,
    ]
    return checkpoint_path, split_datasets


def checkpoint_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("RecBole checkpoint must contain a dictionary.")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise KeyError("RecBole checkpoint is missing state_dict.")
    return state_dict


def validate_checkpoint(checkpoint, config, checkpoint_path):
    expected = {
        "model": str(config["model"]),
        "dataset": str(config["dataset"]),
        "seed": int(config["seed"]),
    }
    actual = {
        "model": get_checkpoint_value(checkpoint, "model"),
        "dataset": get_checkpoint_value(checkpoint, "dataset"),
        "seed": get_checkpoint_value(checkpoint, "seed"),
    }
    if actual["model"] is None or str(actual["model"]).lower() != expected[
        "model"
    ].lower():
        raise ValueError(
            f"Checkpoint model mismatch: expected {expected['model']}, "
            f"got {actual['model']} ({checkpoint_path})."
        )
    if str(actual["dataset"]) != expected["dataset"]:
        raise ValueError(
            f"Checkpoint dataset mismatch: expected {expected['dataset']}, "
            f"got {actual['dataset']} ({checkpoint_path})."
        )
    try:
        actual_seed = int(actual["seed"])
    except (TypeError, ValueError) as error:
        raise ValueError("Checkpoint does not contain a valid seed.") from error
    if actual_seed != expected["seed"]:
        raise ValueError(
            f"Checkpoint seed mismatch: expected {expected['seed']}, "
            f"got {actual_seed} ({checkpoint_path})."
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
    if weight.dim() != 2:
        raise ValueError(
            f"item_embedding.weight must be 2-D, got {tuple(weight.shape)}."
        )
    if weight.size(0) != int(item_num):
        raise ValueError(
            "Embedding row count differs from item_num; automatic truncation is "
            f"forbidden: rows={weight.size(0)}, item_num={item_num}."
        )
    return weight


def is_core_teacher(model):
    return model.__class__.__name__.lower() == "core"


def encode_teacher_state(model, item_seq, item_seq_len):
    """Call each RecBole teacher with its native forward signature."""
    if is_core_teacher(model):
        return model.forward(item_seq)
    return model.forward(item_seq, item_seq_len)


def get_scoring_components(model, item_num):
    """Return item vectors, temperature and the exact full-sort score rule."""
    item_embedding = get_item_embedding(model, item_num)
    if not is_core_teacher(model):
        return item_embedding, 1.0, "state_dot_item_embedding"

    temperature = float(getattr(model, "temperature", 0.0))
    if temperature <= 0:
        raise ValueError(
            f"CORE temperature must be positive, got {temperature}."
        )
    return (
        F.normalize(item_embedding, p=2, dim=-1),
        temperature,
        "l2_normalized_state_dot_l2_normalized_item_div_temperature",
    )


def prepare_state_for_scoring(model, state):
    if is_core_teacher(model):
        return F.normalize(state, p=2, dim=-1)
    return state


def chunked_history_masked_topk(
    state,
    item_embedding,
    history_items,
    topk,
    item_chunk_size,
    score_temperature=1.0,
):
    """Return exact Top-K under state-dot-item scoring without full logits."""
    batch_size, hidden_size = state.shape
    item_num, embedding_size = item_embedding.shape
    if hidden_size != embedding_size:
        raise ValueError(
            f"State dim={hidden_size} differs from item embedding dim="
            f"{embedding_size}."
        )

    best_scores = state.new_empty((batch_size, 0))
    best_items = torch.empty(
        (batch_size, 0), dtype=torch.long, device=state.device
    )

    # Candidate item 0 is padding and is intentionally never scored.
    for start in range(1, item_num, item_chunk_size):
        end = min(start + item_chunk_size, item_num)
        chunk_embedding = item_embedding[start:end]
        chunk_scores = torch.matmul(state, chunk_embedding.transpose(0, 1))
        chunk_scores = chunk_scores / float(score_temperature)

        in_chunk = (history_items >= start) & (history_items < end)
        if torch.any(in_chunk):
            row_ids = torch.arange(
                batch_size, device=state.device
            ).unsqueeze(1).expand_as(history_items)
            chunk_scores[
                row_ids[in_chunk], history_items[in_chunk] - start
            ] = -float("inf")

        chunk_items = torch.arange(
            start, end, dtype=torch.long, device=state.device
        ).unsqueeze(0).expand(batch_size, -1)
        combined_scores = torch.cat((best_scores, chunk_scores), dim=1)
        combined_items = torch.cat((best_items, chunk_items), dim=1)
        running_k = min(topk, combined_scores.size(1))
        best_scores, selected = torch.topk(
            combined_scores,
            k=running_k,
            dim=1,
            largest=True,
            sorted=True,
        )
        best_items = torch.gather(combined_items, 1, selected)

    if best_scores.size(1) != topk:
        raise RuntimeError(
            f"Top-K returned {best_scores.size(1)} items, expected {topk}."
        )
    if not torch.isfinite(best_scores).all():
        raise RuntimeError(
            "Some samples have fewer valid candidates than requested Top-K."
        )
    if torch.any(best_items <= 0):
        raise RuntimeError("Padding item 0 entered exported Top-K candidates.")
    return best_scores, best_items


def make_export_loader(config, split_dataset):
    """Use a non-shuffled training loader to preserve fixed-file row order."""
    return get_dataloader(config, "train")(
        config,
        split_dataset,
        None,
        shuffle=False,
    )


def assert_unique_composite_keys(sample_ids, sequence_lengths, targets):
    seen = set()
    duplicates = []
    for key in zip(
        sample_ids.tolist(),
        sequence_lengths.tolist(),
        targets.tolist(),
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
    topk,
    item_chunk_size,
    checkpoint_path,
    checkpoint_digest,
    output_path,
):
    model.eval()
    device = next(model.parameters()).device
    user_field = config["USER_ID_FIELD"]
    item_field = config["ITEM_ID_FIELD"]
    item_list_field = item_field + config["LIST_SUFFIX"]
    length_field = config["ITEM_LIST_LENGTH_FIELD"]
    item_num = int(split_dataset.item_num)
    effective_topk = min(int(topk), item_num - 1)
    if effective_topk <= 0:
        raise ValueError(f"No non-padding candidate items: item_num={item_num}.")

    item_embedding, score_temperature, score_rule = get_scoring_components(
        model, item_num
    )
    sample_count = len(split_dataset)
    sample_ids_store = torch.empty(sample_count, dtype=torch.long)
    target_items_store = torch.empty(sample_count, dtype=torch.long)
    sequence_lengths_store = torch.empty(sample_count, dtype=torch.long)
    top_items_store = torch.empty(
        (sample_count, effective_topk), dtype=torch.long
    )
    top_scores_store = torch.empty(
        (sample_count, effective_topk), dtype=torch.float32
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
                sequence_lengths = (
                    interaction[length_field].detach().cpu().long()
                )
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
            state = encode_teacher_state(model, item_seq, actual_lengths)
            if state.dim() != 2:
                raise ValueError(
                    f"Teacher forward must return [B,D], got {tuple(state.shape)}."
                )

            scoring_state = prepare_state_for_scoring(model, state)
            top_scores, top_items = chunked_history_masked_topk(
                state=scoring_state,
                item_embedding=item_embedding,
                history_items=item_seq,
                topk=effective_topk,
                item_chunk_size=item_chunk_size,
                score_temperature=score_temperature,
            )

            batch_size = sample_ids.numel()
            end = cursor + batch_size
            if end > sample_count:
                raise RuntimeError("Exporter produced more rows than the split.")
            sample_ids_store[cursor:end] = sample_ids
            target_items_store[cursor:end] = targets
            sequence_lengths_store[cursor:end] = sequence_lengths
            top_items_store[cursor:end] = top_items.cpu()
            top_scores_store[cursor:end] = top_scores.float().cpu()
            cursor = end

    if cursor != sample_count:
        raise RuntimeError(
            f"Exporter wrote {cursor} rows, expected {sample_count}."
        )
    if torch.any(target_items_store <= 0) or torch.any(
        target_items_store >= item_num
    ):
        raise ValueError("target_items contains padding or out-of-range IDs.")
    if torch.any(top_items_store <= 0) or torch.any(
        top_items_store >= item_num
    ):
        raise ValueError("top_items contains padding or out-of-range IDs.")
    assert_unique_composite_keys(
        sample_ids_store,
        sequence_lengths_store,
        target_items_store,
    )

    payload = {
        "sample_ids": sample_ids_store,
        "target_items": target_items_store,
        "sequence_lengths": sequence_lengths_store,
        "top_items": top_items_store,
        "top_scores": top_scores_store,
        "metadata": {
            "format_version": 1,
            "teacher": str(config["model"]),
            "dataset": str(config["dataset"]),
            "phase": phase,
            "seed": int(config["seed"]),
            "topk": effective_topk,
            "requested_topk": int(topk),
            "item_num": item_num,
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "checkpoint_sha256": checkpoint_digest,
            "item_mapping_sha256": item_mapping_sha256(
                split_dataset, item_field
            ),
            "padding_item_id": 0,
            "padding_item_masked": True,
            "history_items_masked": True,
            "history_mask_source": "item_id_list_per_sample",
            "shuffle": False,
            "score_type": "raw_logits_float32",
            "teacher_score_rule": score_rule,
            "score_temperature": score_temperature,
            "state_l2_normalized_for_scoring": is_core_teacher(model),
            "item_l2_normalized_for_scoring": is_core_teacher(model),
            "item_chunk_size": int(item_chunk_size),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(output_path))
    print(f"Saved {config['model']} Top-{effective_topk}: {output_path}")
    print(f"sample_ids shape: {tuple(sample_ids_store.shape)}")
    print(f"top_items shape: {tuple(top_items_store.shape)}")
    print(f"top_scores shape: {tuple(top_scores_store.shape)}")
    print(
        "item_mapping_sha256: "
        f"{payload['metadata']['item_mapping_sha256']}"
    )


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

    split_datasets = None
    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Explicit checkpoint does not exist: {checkpoint_path}"
            )
    else:
        try:
            checkpoint_path = find_unique_checkpoint(
                config,
                args.teacher_model,
                args.dataset,
                args.seed,
            )
        except FileNotFoundError:
            if not args.train_if_missing:
                raise
            dataset = create_dataset(config)
            checkpoint_path, split_datasets = train_teacher(
                config, dataset, logger
            )

    if split_datasets is None:
        dataset = create_dataset(config)
        split_datasets = dataset.build()
    if len(split_datasets) != 3:
        raise RuntimeError(
            f"Expected three benchmark splits, got {len(split_datasets)}."
        )
    expected_phases = list(config["benchmark_filename"])
    if expected_phases != list(BENCHMARK_PHASES):
        raise RuntimeError(
            f"Unexpected benchmark order: {expected_phases}."
        )

    model = load_model(config, split_datasets[0], checkpoint_path)
    checkpoint_digest = file_sha256(checkpoint_path)
    output_dir = Path(args.output_dir)
    for phase in args.split:
        split_index = BENCHMARK_PHASES.index(phase)
        output_path = output_dir / (
            f"{args.teacher_model}_top{args.topk}_{args.dataset}_"
            f"{phase}_seed{args.seed}.pt"
        )
        export_split(
            model=model,
            split_dataset=split_datasets[split_index],
            config=config,
            phase=phase,
            topk=args.topk,
            item_chunk_size=args.item_chunk_size,
            checkpoint_path=checkpoint_path,
            checkpoint_digest=checkpoint_digest,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()
