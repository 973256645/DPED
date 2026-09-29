"""Export MBHT/DPED-compatible unified four-prefix logits from RecBole.

For each sample, this script runs SASRec, GRU4Rec, FEARec or CORE on four independently
truncated causal prefixes.  It obtains an exact history-masked Top-K for each
prefix in item chunks, builds the sorted union of those candidates, and stores
native teacher logits for every prefix on that shared candidate space.
Fixed benchmark row order is preserved and a missing checkpoint is never
trained implicitly.
"""

import argparse
import hashlib
from logging import getLogger
import os
from pathlib import Path

import torch
import torch.nn.functional as F

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
        description="Export unified four-prefix logits from a RecBole teacher."
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
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--output_dir", default="teacher_unified_prefix_logits")
    parser.add_argument("--item_chunk_size", type=int, default=10000)
    parser.add_argument(
        "--score_dtype", choices=("float16", "float32"), default="float16"
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


def get_scoring_components(model, item_num):
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
    transition_mask = prefix_lengths[:, 1:] > prefix_lengths[:, :-1]
    return prefix_lengths, prefix_mask, transition_mask


def truncate_batch(item_seq, prefix_lengths):
    positions = torch.arange(
        item_seq.size(1), device=item_seq.device
    ).unsqueeze(0)
    return item_seq.masked_fill(positions >= prefix_lengths.unsqueeze(1), 0)


def chunked_history_masked_topk(
    state,
    item_embedding,
    history_items,
    topk,
    item_chunk_size,
    score_temperature=1.0,
):
    """Return exact Top-K without materializing [batch_size, item_num]."""
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
    for start in range(1, item_num, item_chunk_size):
        end = min(start + item_chunk_size, item_num)
        chunk_scores = torch.matmul(
            state, item_embedding[start:end].transpose(0, 1)
        )
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
            combined_scores, k=running_k, dim=1, largest=True, sorted=True
        )
        best_items = torch.gather(combined_items, 1, selected)
    if best_scores.size(1) != topk or not torch.isfinite(best_scores).all():
        raise RuntimeError(
            "Some prefixes have fewer valid candidates than requested Top-K."
        )
    if torch.any(best_items <= 0):
        raise RuntimeError("Padding item 0 entered exported Top-K candidates.")
    return best_scores, best_items


def build_sorted_candidate_union(prefix_top_items):
    if prefix_top_items.dim() != 3:
        raise ValueError("prefix_top_items must have shape [B, P, K].")
    batch_size = prefix_top_items.size(0)
    flattened = prefix_top_items.reshape(batch_size, -1)
    sorted_items = torch.sort(flattened, dim=1).values
    is_new = torch.ones_like(sorted_items, dtype=torch.bool)
    is_new[:, 1:] = sorted_items[:, 1:] != sorted_items[:, :-1]
    destinations = torch.cumsum(is_new.long(), dim=1) - 1
    candidate_items = torch.zeros_like(sorted_items)
    candidate_items.scatter_(1, destinations, sorted_items)
    union_sizes = is_new.sum(dim=1)
    candidate_mask = torch.arange(
        candidate_items.size(1), device=candidate_items.device
    ).unsqueeze(0) < union_sizes.unsqueeze(1)
    candidate_items = candidate_items.masked_fill(~candidate_mask, 0)
    return candidate_items, candidate_mask, union_sizes


def validate_native_readout_once(
    model,
    interaction,
    full_state,
    item_embedding,
    item_num,
    score_temperature,
):
    native_scores = model.full_sort_predict(interaction).view(-1, item_num)
    scoring_state = prepare_state_for_scoring(model, full_state)
    manual_scores = torch.matmul(
        scoring_state, item_embedding.transpose(0, 1)
    ) / float(score_temperature)
    if native_scores.shape != manual_scores.shape:
        raise RuntimeError("Manual and native teacher score shapes differ.")
    if not torch.allclose(native_scores, manual_scores, rtol=1e-4, atol=1e-5):
        max_error = (native_scores - manual_scores).abs().max().item()
        raise RuntimeError(
            "RecBole native readout validation failed; maximum absolute "
            f"error is {max_error:.6g}."
        )


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
    requested_topk,
    item_chunk_size,
    score_dtype,
):
    model.eval()
    device = next(model.parameters()).device
    user_field = config["USER_ID_FIELD"]
    item_field = config["ITEM_ID_FIELD"]
    item_list_field = item_field + config["LIST_SUFFIX"]
    length_field = config["ITEM_LIST_LENGTH_FIELD"]
    item_num = int(split_dataset.item_num)
    item_embedding, score_temperature, score_rule = get_scoring_components(
        model, item_num
    )
    effective_topk = min(int(requested_topk), item_num - 1)
    if effective_topk <= 0:
        raise ValueError(f"No non-padding candidate items: item_num={item_num}.")
    prefix_count = len(PREFIX_RATIOS)
    max_union_size = prefix_count * effective_topk
    stored_dtype = torch.float16 if score_dtype == "float16" else torch.float32

    sample_count = len(split_dataset)
    sample_ids_store = torch.empty(sample_count, dtype=torch.long)
    targets_store = torch.empty(sample_count, dtype=torch.long)
    sequence_lengths_store = torch.empty(sample_count, dtype=torch.long)
    prefix_lengths_store = torch.empty((sample_count, 4), dtype=torch.long)
    prefix_mask_store = torch.empty((sample_count, 4), dtype=torch.bool)
    transition_mask_store = torch.empty((sample_count, 3), dtype=torch.bool)
    candidate_items_store = torch.empty(
        (sample_count, max_union_size), dtype=torch.int32
    )
    teacher_logits_store = torch.empty(
        (sample_count, 4, max_union_size), dtype=stored_dtype
    )

    cursor = 0
    readout_verified = False
    union_size_sum = 0
    union_size_min = max_union_size
    union_size_max = 0
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
            prefix_lengths, prefix_mask, transition_mask = four_prefix_lengths(
                actual_lengths
            )

            prefix_states = []
            prefix_histories = []
            for prefix_index in range(prefix_count):
                current_lengths = prefix_lengths[:, prefix_index]
                prefix_items = truncate_batch(item_seq, current_lengths)
                state = encode_teacher_state(
                    model, prefix_items, current_lengths
                )
                if state.dim() != 2 or state.size(1) != item_embedding.size(1):
                    raise ValueError(
                        "Teacher forward must return [B, embedding_dim]; got "
                        f"{tuple(state.shape)}."
                    )
                prefix_states.append(state)
                prefix_histories.append(prefix_items)
            prefix_states = torch.stack(prefix_states, dim=1)
            if not torch.isfinite(prefix_states).all():
                raise RuntimeError("Teacher prefix states contain NaN or Inf.")

            if not readout_verified:
                validate_native_readout_once(
                    model=model,
                    interaction=interaction,
                    full_state=prefix_states[:, -1],
                    item_embedding=item_embedding,
                    item_num=item_num,
                    score_temperature=score_temperature,
                )
                readout_verified = True

            prefix_top_items = []
            for prefix_index in range(prefix_count):
                _, top_items = chunked_history_masked_topk(
                    state=prepare_state_for_scoring(
                        model, prefix_states[:, prefix_index]
                    ),
                    item_embedding=item_embedding,
                    history_items=prefix_histories[prefix_index],
                    topk=effective_topk,
                    item_chunk_size=item_chunk_size,
                    score_temperature=score_temperature,
                )
                prefix_top_items.append(top_items)
            prefix_top_items = torch.stack(prefix_top_items, dim=1)

            candidate_items, candidate_mask, union_sizes = (
                build_sorted_candidate_union(prefix_top_items)
            )
            candidate_embeddings = item_embedding[candidate_items]
            scoring_prefix_states = prepare_state_for_scoring(
                model, prefix_states
            )
            teacher_logits = torch.einsum(
                "bph,buh->bpu", scoring_prefix_states, candidate_embeddings
            ) / float(score_temperature)
            teacher_logits = teacher_logits.masked_fill(
                ~candidate_mask.unsqueeze(1), 0.0
            )
            active_mask = candidate_mask.unsqueeze(1).expand_as(teacher_logits)
            if not torch.isfinite(teacher_logits[active_mask]).all():
                raise RuntimeError("Unified teacher logits contain NaN or Inf.")

            batch_size = sample_ids.numel()
            end = cursor + batch_size
            if end > sample_count:
                raise RuntimeError("Exporter produced more rows than the split.")
            sample_ids_store[cursor:end] = sample_ids
            targets_store[cursor:end] = targets
            sequence_lengths_store[cursor:end] = sequence_lengths
            prefix_lengths_store[cursor:end] = prefix_lengths.cpu()
            prefix_mask_store[cursor:end] = prefix_mask.cpu()
            transition_mask_store[cursor:end] = transition_mask.cpu()
            candidate_items_store[cursor:end] = candidate_items.to(
                device="cpu", dtype=torch.int32
            )
            stored_logits = teacher_logits.to(device="cpu", dtype=stored_dtype)
            if not torch.isfinite(stored_logits).all():
                raise RuntimeError(
                    "Teacher logits overflowed in the selected storage dtype."
                )
            teacher_logits_store[cursor:end] = stored_logits
            cursor = end

            union_sizes_cpu = union_sizes.cpu()
            union_size_sum += int(union_sizes_cpu.sum().item())
            union_size_min = min(
                union_size_min, int(union_sizes_cpu.min().item())
            )
            union_size_max = max(
                union_size_max, int(union_sizes_cpu.max().item())
            )

    if cursor != sample_count:
        raise RuntimeError(
            f"Exporter wrote {cursor} rows, expected {sample_count}."
        )
    if torch.any(targets_store <= 0) or torch.any(targets_store >= item_num):
        raise ValueError("target_items contains padding or out-of-range IDs.")
    assert_unique_composite_keys(
        sample_ids_store, sequence_lengths_store, targets_store
    )
    mean_union_size = union_size_sum / float(sample_count)

    payload = {
        "sample_ids": sample_ids_store,
        "target_items": targets_store,
        "sequence_lengths": sequence_lengths_store,
        "prefix_lengths": prefix_lengths_store,
        "prefix_mask": prefix_mask_store,
        "transition_mask": transition_mask_store,
        "candidate_items": candidate_items_store,
        "teacher_logits": teacher_logits_store,
        "metadata": {
            "format_version": 1,
            "teacher": str(config["model"]),
            "dataset": str(config["dataset"]),
            "phase": phase,
            "seed": int(config["seed"]),
            "prefix_ratios": list(PREFIX_RATIOS),
            "num_prefixes": prefix_count,
            "topk_per_prefix": effective_topk,
            "requested_topk": int(requested_topk),
            "max_union_size": max_union_size,
            "mean_union_size": mean_union_size,
            "min_union_size": union_size_min,
            "max_observed_union_size": union_size_max,
            "item_num": item_num,
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "checkpoint_sha256": checkpoint_digest,
            "item_mapping_sha256": item_mapping_sha256(
                split_dataset, item_field
            ),
            "padding_item_id": 0,
            "causal_prefix_export": True,
            "unified_candidate_space": True,
            "candidate_construction": (
                "union_of_history_masked_topk_from_four_prefixes"
            ),
            "prefix_definition": "ceil(L/4),ceil(L/2),ceil(3L/4),L",
            "short_sequence_duplicates_masked": True,
            "transition_mask_stored": True,
            "candidate_padding_item_id": 0,
            "candidate_items_sorted": True,
            "candidate_mask_definition": "candidate_items != 0",
            "teacher_logits_are_raw": True,
            "teacher_logit_definition": (
                f"{config['model']} native score rule on unified candidates"
            ),
            "teacher_score_rule": score_rule,
            "score_temperature": score_temperature,
            "state_l2_normalized_for_scoring": is_core_teacher(model),
            "item_l2_normalized_for_scoring": is_core_teacher(model),
            "topk_selection_padding_masked": True,
            "topk_selection_prefix_history_masked": True,
            "item_id_dtype": "int32",
            "score_dtype": score_dtype,
            "native_readout_verified": readout_verified,
            "item_chunk_size": int(item_chunk_size),
            "shuffle": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(output_path))
    print(f"Saved unified four-prefix teacher logits: {output_path}")
    print(f"sample_ids shape: {tuple(sample_ids_store.shape)}")
    print(f"candidate_items shape: {tuple(candidate_items_store.shape)}")
    print(f"teacher_logits shape: {tuple(teacher_logits_store.shape)}")
    print(
        "union size min/mean/max: "
        f"{union_size_min}/{mean_union_size:.2f}/{union_size_max}"
    )
    print(f"candidate_items dtype: {candidate_items_store.dtype}")
    print(f"teacher_logits dtype: {teacher_logits_store.dtype}")


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
            f"{args.teacher_model}_unified_prefix_top{args.topk}_trajectory_"
            f"{args.dataset}_{phase}_seed{args.seed}.pt"
        )
        export_split(
            model=model,
            split_dataset=split_datasets[split_index],
            config=config,
            phase=phase,
            checkpoint_path=checkpoint_path,
            checkpoint_digest=checkpoint_digest,
            output_path=output_path,
            requested_topk=args.topk,
            item_chunk_size=args.item_chunk_size,
            score_dtype=args.score_dtype,
        )


if __name__ == "__main__":
    main()
