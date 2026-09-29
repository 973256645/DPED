# -*- coding: utf-8 -*-
"""Efficient SASRec trained with RCE-KD and DPED.

This module preserves the original RecBole SASRec encoder and extends only
its training objective.  The objective contains:

* ordinary next-item cross entropy;
* final-state Rejuvenated Cross-Entropy KD (RCE-KD);
* four-prefix state trajectory matching;
* unified-prefix distribution matching;
* optional adjacent direction and magnitude trajectory matching.

The implementation reuses the RCE-KD mixin in ``sasrecrcekd.py`` and the
same DPED helper modules as the P3-State2 project.  Put these two existing
DPED files either in the RecBole project root or beside this module:

* direction_magnitude_trajectory_loss.py
* teacher_unified_prefix_topk_loader_SharedTeacher.py
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from recbole.model.sequential_recommender.sasrec import SASRec
from recbole.model.sequential_recommender.sasrecrcekd import (
    _RCEKDStudentMixin,
)

try:
    from recbole.model.sequential_recommender.direction_magnitude_trajectory_loss import (
        DirectionMagnitudeTrajectoryLoss,
    )
    from recbole.model.sequential_recommender.teacher_unified_prefix_topk_loader_SharedTeacher import (
        TeacherUnifiedPrefixTopKStoreSharedTeacher,
    )
except ModuleNotFoundError:
    # Compatibility with the current P3 project, where shared helpers live in
    # the repository root rather than inside the RecBole package.
    from direction_magnitude_trajectory_loss import (
        DirectionMagnitudeTrajectoryLoss,
    )
    from teacher_unified_prefix_topk_loader_SharedTeacher import (
        TeacherUnifiedPrefixTopKStoreSharedTeacher,
    )


class SASRecRCEDPED(_RCEKDStudentMixin, SASRec):
    """Original SASRec optimized with CE, RCE-KD, and DPED losses."""

    def __init__(self, config, dataset):
        SASRec.__init__(self, config, dataset)
        self._init_rce_kd(config, dataset)

        if not self.rce_kd_enabled:
            raise ValueError("SASRecRCEDPED requires rce_kd_enabled=True.")
        if self.teacher_topk_store is None:
            raise ValueError("RCE-DPED requires a final teacher Top-K store.")
        if self.teacher_trajectory_store is None:
            raise ValueError("RCE-DPED requires a teacher trajectory store.")

        self.trajectory_enabled = bool(config["trajectory_enabled"])
        self.trajectory_weight = float(config["trajectory_weight"])
        self.trajectory_normalize = bool(config["trajectory_normalize"])

        self.prefix_distribution_enabled = bool(
            config["prefix_distribution_enabled"]
        )
        self.prefix_distribution_topk = int(
            config["prefix_distribution_topk"]
        )
        self.prefix_distribution_temperature = float(
            config["prefix_distribution_temperature"]
        )
        self.prefix_distribution_weight = float(
            config["prefix_distribution_weight"]
        )

        self.direction_magnitude_enabled = bool(
            config["direction_magnitude_enabled"]
        )
        self.evolution_temperature = float(config["evolution_temperature"])
        self.direction_trajectory_weight = float(
            config["direction_trajectory_weight"]
        )
        self.magnitude_trajectory_weight = float(
            config["magnitude_trajectory_weight"]
        )
        self.magnitude_smooth_l1_beta = float(
            config["magnitude_smooth_l1_beta"]
        )

        self._validate_dped_config()

        self.trajectory_projector = nn.Linear(
            self.hidden_size,
            self.teacher_trajectory_store.hidden_size,
            bias=False,
        )
        nn.init.xavier_uniform_(self.trajectory_projector.weight)

        need_unified_prefix = (
            self.prefix_distribution_enabled
            or self.direction_magnitude_enabled
        )
        self.teacher_unified_prefix_store = None
        self.direction_magnitude_loss_fct = None
        if need_unified_prefix:
            self.teacher_unified_prefix_store = (
                TeacherUnifiedPrefixTopKStoreSharedTeacher(
                    file_path=config["teacher_unified_prefix_topk_path"],
                    dataset=dataset,
                    config=config,
                    expected_phase="train",
                    expected_topk=self.prefix_distribution_topk,
                )
            )
            self.direction_magnitude_loss_fct = (
                DirectionMagnitudeTrajectoryLoss(
                    prefix_temperature=(
                        self.prefix_distribution_temperature
                    ),
                    evolution_temperature=self.evolution_temperature,
                    magnitude_smooth_l1_beta=(
                        self.magnitude_smooth_l1_beta
                    ),
                )
            )
            self._validate_dped_teacher_sources(config)

        self.last_trajectory_loss = None
        self.last_prefix_distribution_loss = None
        self.last_direction_trajectory_loss = None
        self.last_magnitude_trajectory_loss = None
        self.last_direction_magnitude_diagnostics = None
        self.last_efficiency_diagnostics = None

    def _validate_dped_config(self):
        if self.trajectory_weight < 0:
            raise ValueError("trajectory_weight cannot be negative.")
        if self.prefix_distribution_topk <= 0:
            raise ValueError("prefix_distribution_topk must be positive.")
        if self.prefix_distribution_temperature <= 0:
            raise ValueError(
                "prefix_distribution_temperature must be positive."
            )
        if self.evolution_temperature <= 0:
            raise ValueError("evolution_temperature must be positive.")
        if self.magnitude_smooth_l1_beta <= 0:
            raise ValueError("magnitude_smooth_l1_beta must be positive.")
        if (
            self.prefix_distribution_weight < 0
            or self.direction_trajectory_weight < 0
            or self.magnitude_trajectory_weight < 0
        ):
            raise ValueError("DPED loss weights cannot be negative.")
        if (
            self.prefix_distribution_enabled
            or self.direction_magnitude_enabled
        ) and not self.trajectory_enabled:
            raise ValueError(
                "Prefix/direction DPED requires trajectory_enabled=True."
            )
        if self.prefix_distribution_topk != self.teacher_topk_store.topk:
            raise ValueError(
                "RCE-KD and DPED must use the same teacher Top-K width."
            )

    def _validate_dped_teacher_sources(self, config):
        stores = {
            "final Top-K": self.teacher_topk_store,
            "state trajectory": self.teacher_trajectory_store,
            "unified prefix logits": self.teacher_unified_prefix_store,
        }
        expected_seed = int(config["teacher_seed"])
        expected_dataset = str(config["dataset"])
        checkpoints = {}

        for source_name, store in stores.items():
            if store is None:
                raise ValueError(
                    "Missing required teacher source: {}.".format(source_name)
                )
            metadata = getattr(store, "metadata", None)
            if not isinstance(metadata, dict):
                raise ValueError(
                    "{} metadata is unavailable.".format(source_name)
                )
            if int(metadata.get("seed", -1)) != expected_seed:
                raise ValueError(
                    "{} seed does not match teacher_seed={}.".format(
                        source_name, expected_seed
                    )
                )
            if str(metadata.get("dataset", "")) != expected_dataset:
                raise ValueError(
                    "{} dataset does not match dataset='{}'.".format(
                        source_name, expected_dataset
                    )
                )
            checkpoint = str(metadata.get("checkpoint", ""))
            if not checkpoint:
                raise ValueError(
                    "{} metadata is missing checkpoint.".format(source_name)
                )
            checkpoints[source_name] = Path(checkpoint).name

        if len(set(checkpoints.values())) != 1:
            raise ValueError(
                "Teacher artifacts come from different checkpoints: {}".format(
                    checkpoints
                )
            )
        if (
            self.teacher_topk_store.topk
            != self.teacher_unified_prefix_store.topk
        ):
            raise ValueError(
                "Final and unified-prefix stores use different Top-K sizes."
            )
        if len({len(store) for store in stores.values()}) != 1:
            raise ValueError(
                "Teacher sources contain different numbers of samples."
            )

        reference_store = self.teacher_topk_store
        for source_name, store in stores.items():
            if store is reference_store:
                continue
            aligned_fields = (
                ("sample_ids", reference_store.sample_ids, store.sample_ids),
                (
                    "sequence_lengths",
                    reference_store.sequence_lengths,
                    store.sequence_lengths,
                ),
                (
                    "target_items",
                    reference_store.target_items,
                    store.target_items,
                ),
            )
            for field_name, reference_value, source_value in aligned_fields:
                if not torch.equal(reference_value, source_value):
                    raise ValueError(
                        "{} {} are not aligned with final Top-K.".format(
                            source_name, field_name
                        )
                    )

        # These tensors are immutable CPU artifacts.  Validate their alignment
        # once here instead of synchronizing the GPU on every training batch.
        trajectory_store = self.teacher_trajectory_store
        unified_store = self.teacher_unified_prefix_store
        if not torch.equal(
            trajectory_store.prefix_lengths,
            unified_store.prefix_lengths,
        ):
            raise ValueError(
                "State and unified prediction files use different prefixes."
            )
        if not torch.equal(
            trajectory_store.prefix_mask,
            unified_store.prefix_mask,
        ):
            raise ValueError(
                "State and unified prediction files use different prefix masks."
            )
        if not torch.equal(
            trajectory_store.prefix_lengths[:, -1],
            trajectory_store.sequence_lengths,
        ):
            raise ValueError(
                "The fourth teacher prefix is not the full sequence."
            )

    def _encode_all_positions(self, item_seq):
        """Run SASRec once and retain every causal position state.

        SASRec uses a causal attention mask, so the state at position ``t``
        cannot access items after ``t``.  Consequently, gathering four prefix
        endpoints from this tensor is equivalent to four truncated inference
        passes while avoiding three redundant Transformer executions.  During
        training it also keeps all prefix states under one dropout realization.
        """
        position_ids = torch.arange(
            item_seq.size(1),
            dtype=torch.long,
            device=item_seq.device,
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        input_emb = self.item_embedding(item_seq) + self.position_embedding(
            position_ids
        )
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_seq)
        encoded_layers = self.trm_encoder(
            input_emb,
            extended_attention_mask,
            output_all_encoded_layers=True,
        )
        return encoded_layers[-1]

    @staticmethod
    def _gather_prefix_states(encoded_sequence, prefix_lengths):
        """Gather [B, P, H] endpoint states from one [B, L, H] encoding."""
        max_length = encoded_sequence.size(1)
        endpoint_indices = prefix_lengths.clamp(
            min=1,
            max=max_length,
        ) - 1
        gather_indices = endpoint_indices.unsqueeze(-1).expand(
            -1,
            -1,
            encoded_sequence.size(-1),
        )
        return encoded_sequence.gather(1, gather_indices)

    def _prefix_distribution_loss_only(
        self,
        student_logits,
        teacher_logits,
        prefix_candidate_mask,
        auxiliary_prefix_mask,
    ):
        """Compute only the active prefix KL, without evolution diagnostics."""
        if torch.any(prefix_candidate_mask.sum(dim=-1) == 0):
            raise ValueError("A masked prefix distribution has no candidates.")
        temperature = self.prefix_distribution_temperature
        teacher_scaled = teacher_logits.detach().to(torch.float32) / temperature
        student_scaled = student_logits.to(torch.float32) / temperature
        teacher_log_prob = F.log_softmax(
            teacher_scaled.masked_fill(~prefix_candidate_mask, -1.0e9),
            dim=-1,
        )
        student_log_prob = F.log_softmax(
            student_scaled.masked_fill(~prefix_candidate_mask, -1.0e9),
            dim=-1,
        )
        teacher_prob = teacher_log_prob.exp()
        kl_terms = teacher_prob * (teacher_log_prob - student_log_prob)
        kl_terms = torch.where(
            prefix_candidate_mask,
            kl_terms,
            torch.zeros_like(kl_terms),
        )
        point_kl = kl_terms.sum(dim=-1) * (temperature ** 2)
        weights = auxiliary_prefix_mask.to(dtype=point_kl.dtype)
        loss = (point_kl * weights).sum() / weights.sum().clamp_min(1.0)
        diagnostics = {
            "auxiliary_prefix_kl": loss.detach(),
            "fast_prefix_only_path": loss.detach().new_ones(()),
        }
        return loss, diagnostics

    @staticmethod
    def _prefix_candidate_availability(
        item_seq,
        prefix_lengths,
        candidate_items,
        candidate_mask,
        item_num,
    ):
        """Mask unified candidates already observed at each causal prefix."""
        if candidate_items.size(1) == 0:
            raise ValueError("Unified candidate width cannot be zero.")

        searchable_items = candidate_items.masked_fill(
            ~candidate_mask,
            item_num,
        )
        insertion_positions = torch.searchsorted(
            searchable_items,
            item_seq.contiguous(),
        )
        safe_positions = insertion_positions.clamp_max(
            candidate_items.size(1) - 1
        )
        located_items = candidate_items.gather(1, safe_positions)
        located_valid = candidate_mask.gather(1, safe_positions)
        history_item_in_union = (
            (item_seq != 0)
            & located_valid
            & (located_items == item_seq)
            & (insertion_positions < candidate_items.size(1))
        )

        sequence_positions = torch.arange(
            item_seq.size(1), device=item_seq.device
        ).unsqueeze(0)
        availability = []
        for prefix_index in range(prefix_lengths.size(1)):
            included_history = history_item_in_union & (
                sequence_positions
                < prefix_lengths[:, prefix_index].unsqueeze(1)
            )
            seen_counts = torch.zeros_like(
                candidate_items,
                dtype=torch.int32,
            )
            seen_counts.scatter_add_(
                dim=1,
                index=safe_positions,
                src=included_history.to(torch.int32),
            )
            availability.append(candidate_mask & (seen_counts == 0))
        return torch.stack(availability, dim=1)

    def _rce_final_loss(
        self,
        student_logits,
        item_seq,
        pos_items,
        teacher_top_items,
        teacher_top_scores,
        teacher_final_state,
    ):
        if not self.rce_kd_enabled or self.rce_kd_weight == 0:
            self.last_rce_kd_diagnostics = {}
            return student_logits.new_zeros(())

        loss = self.rce_kd_loss_fct(
            student_logits=student_logits,
            teacher_state=teacher_final_state,
            teacher_top_items=teacher_top_items,
            teacher_top_scores=teacher_top_scores,
            history_items=item_seq,
            target_items=pos_items,
        )
        self.last_rce_kd_diagnostics = dict(
            self.rce_kd_loss_fct.last_diagnostics
        )
        return loss

    def _combined_trajectory_losses(
        self,
        item_seq,
        student_states,
        state_prefix_mask,
        teacher_states,
        unified_batch,
    ):
        need_state = self.trajectory_enabled and self.trajectory_weight > 0
        need_prefix = (
            self.prefix_distribution_enabled
            and self.prefix_distribution_weight > 0
        )
        need_direction = (
            self.direction_magnitude_enabled
            and self.direction_trajectory_weight > 0
        )
        need_magnitude = (
            self.direction_magnitude_enabled
            and self.magnitude_trajectory_weight > 0
        )
        need_prediction = need_prefix or need_direction or need_magnitude
        zero = student_states.new_zeros(())
        if not need_state and not need_prediction:
            return zero, zero, zero, zero, None

        state_loss = zero
        if need_state:
            projected_states = self.trajectory_projector(student_states)
            point_teacher_states = teacher_states
            if self.trajectory_normalize:
                hidden_size = projected_states.size(-1)
                projected_states = F.layer_norm(
                    projected_states, (hidden_size,)
                )
                point_teacher_states = F.layer_norm(
                    point_teacher_states, (hidden_size,)
                )
            point_losses = F.mse_loss(
                projected_states,
                point_teacher_states,
                reduction="none",
            ).mean(dim=-1)
            valid_points = state_prefix_mask.to(dtype=point_losses.dtype)
            state_loss = (point_losses * valid_points).sum() / (
                valid_points.sum().clamp_min(1.0)
            )

        prefix_loss = zero
        direction_loss = zero
        magnitude_loss = zero
        diagnostics = None
        if need_prediction:
            if unified_batch is None:
                raise RuntimeError(
                    "Unified-prefix teacher data was not preloaded."
                )
            (
                unified_prefix_lengths,
                unified_prefix_mask,
                transition_mask,
                candidate_items,
                candidate_mask,
                teacher_logits,
            ) = unified_batch
            candidate_embeddings = self.item_embedding(candidate_items)
            student_logits = torch.einsum(
                "bph,buh->bpu",
                student_states,
                candidate_embeddings,
            )
            prefix_candidate_mask = self._prefix_candidate_availability(
                item_seq=item_seq,
                prefix_lengths=unified_prefix_lengths,
                candidate_items=candidate_items,
                candidate_mask=candidate_mask,
                item_num=self.n_items,
            )
            auxiliary_prefix_mask = unified_prefix_mask & (
                unified_prefix_lengths
                < unified_prefix_lengths[:, -1].unsqueeze(1)
            )
            auxiliary_prefix_mask[:, -1] = False

            # The main experiments use direction=magnitude=0.  Avoid computing
            # three transition losses, Top-10 diagnostics, and their autograd
            # graphs when only the prefix-distribution KL contributes.
            if need_prefix and not need_direction and not need_magnitude:
                prefix_loss, diagnostics = (
                    self._prefix_distribution_loss_only(
                        student_logits=student_logits,
                        teacher_logits=teacher_logits,
                        prefix_candidate_mask=prefix_candidate_mask,
                        auxiliary_prefix_mask=auxiliary_prefix_mask,
                    )
                )
            else:
                (
                    prefix_loss,
                    direction_loss,
                    magnitude_loss,
                    diagnostics,
                ) = self.direction_magnitude_loss_fct(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    candidate_items=candidate_items,
                    candidate_mask=candidate_mask,
                    prefix_candidate_mask=prefix_candidate_mask,
                    prefix_mask=unified_prefix_mask,
                    auxiliary_prefix_mask=auxiliary_prefix_mask,
                    transition_mask=transition_mask,
                )

        return (
            state_loss,
            prefix_loss,
            direction_loss,
            magnitude_loss,
            diagnostics,
        )

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        sample_ids = interaction[self.rce_kd_sample_id_field]
        device = self.item_embedding.weight.device

        need_rce = self.rce_kd_enabled and self.rce_kd_weight > 0
        need_state = self.trajectory_enabled and self.trajectory_weight > 0
        need_prefix = (
            self.prefix_distribution_enabled
            and self.prefix_distribution_weight > 0
        )
        need_direction = (
            self.direction_magnitude_enabled
            and self.direction_trajectory_weight > 0
        )
        need_magnitude = (
            self.direction_magnitude_enabled
            and self.magnitude_trajectory_weight > 0
        )
        need_prediction = need_prefix or need_direction or need_magnitude
        need_prefix_states = need_state or need_prediction

        teacher_top_items = None
        teacher_top_scores = None
        if need_rce:
            teacher_top_items, teacher_top_scores = (
                self.teacher_topk_store.get_batch(
                    sample_ids=sample_ids,
                    sequence_lengths=item_seq_len,
                    target_items=pos_items,
                    device=device,
                    validate_targets=self.rce_kd_validate_targets,
                )
            )

        state_prefix_lengths = None
        state_prefix_mask = None
        teacher_states = None
        if need_rce or need_state:
            (
                state_prefix_lengths,
                state_prefix_mask,
                teacher_states,
            ) = self.teacher_trajectory_store.get_batch(
                sample_ids=sample_ids,
                sequence_lengths=item_seq_len,
                target_items=pos_items,
                device=device,
            )

        unified_batch = None
        if need_prediction:
            unified_batch = self.teacher_unified_prefix_store.get_batch(
                sample_ids=sample_ids,
                sequence_lengths=item_seq_len,
                target_items=pos_items,
                device=device,
            )

        # One causal Transformer pass supplies both the final prediction state
        # and all four prefix endpoint states.
        encoded_sequence = self._encode_all_positions(item_seq)
        full_sequence_state = self.gather_indexes(
            encoded_sequence,
            item_seq_len - 1,
        )
        prefix_lengths = state_prefix_lengths
        if prefix_lengths is None and unified_batch is not None:
            prefix_lengths = unified_batch[0]
        if state_prefix_mask is None and unified_batch is not None:
            state_prefix_mask = unified_batch[1]
        student_states = None
        if need_prefix_states:
            if prefix_lengths is None:
                raise RuntimeError("Prefix lengths were not preloaded.")
            student_states = self._gather_prefix_states(
                encoded_sequence,
                prefix_lengths,
            )

        student_logits = torch.matmul(
            full_sequence_state,
            self.item_embedding.weight.transpose(0, 1),
        )
        rec_loss = self.loss_fct(student_logits, pos_items)
        rce_kd_loss = self._rce_final_loss(
            student_logits=student_logits,
            item_seq=item_seq,
            pos_items=pos_items,
            teacher_top_items=teacher_top_items,
            teacher_top_scores=teacher_top_scores,
            teacher_final_state=(
                None if teacher_states is None else teacher_states[:, -1, :]
            ),
        )
        (
            state_trajectory_loss,
            prefix_distribution_loss,
            direction_trajectory_loss,
            magnitude_trajectory_loss,
            diagnostics,
        ) = self._combined_trajectory_losses(
            item_seq=item_seq,
            student_states=(
                student_states
                if student_states is not None
                else full_sequence_state.unsqueeze(1)
            ),
            state_prefix_mask=state_prefix_mask,
            teacher_states=teacher_states,
            unified_batch=unified_batch,
        )

        self.last_rec_loss = rec_loss.detach()
        self.last_rce_kd_loss = rce_kd_loss.detach()
        self.last_trajectory_loss = state_trajectory_loss.detach()
        self.last_prefix_distribution_loss = (
            prefix_distribution_loss.detach()
        )
        self.last_direction_trajectory_loss = (
            direction_trajectory_loss.detach()
        )
        self.last_magnitude_trajectory_loss = (
            magnitude_trajectory_loss.detach()
        )
        self.last_direction_magnitude_diagnostics = diagnostics
        self.last_efficiency_diagnostics = {
            "student_encoder_forward_count": 1.0,
            "avoided_prefix_forward_count": (
                3.0 if need_prefix_states else 0.0
            ),
            "teacher_trajectory_fetch_count": (
                1.0 if (need_rce or need_state) else 0.0
            ),
            "teacher_topk_fetch_count": 1.0 if need_rce else 0.0,
            "teacher_unified_fetch_count": (
                1.0 if need_prediction else 0.0
            ),
            "single_pass_prefix_extraction": (
                1.0 if need_prefix_states else 0.0
            ),
            "fast_prefix_only_path": (
                1.0
                if need_prefix and not need_direction and not need_magnitude
                else 0.0
            ),
        }

        losses = [rec_loss]
        if self.rce_kd_enabled and self.rce_kd_weight > 0:
            losses.append(self.rce_kd_weight * rce_kd_loss)
        if self.trajectory_enabled and self.trajectory_weight > 0:
            losses.append(self.trajectory_weight * state_trajectory_loss)
        if (
            self.prefix_distribution_enabled
            and self.prefix_distribution_weight > 0
        ):
            losses.append(
                self.prefix_distribution_weight * prefix_distribution_loss
            )
        if (
            self.direction_magnitude_enabled
            and self.direction_trajectory_weight > 0
        ):
            losses.append(
                self.direction_trajectory_weight * direction_trajectory_loss
            )
        if (
            self.direction_magnitude_enabled
            and self.magnitude_trajectory_weight > 0
        ):
            losses.append(
                self.magnitude_trajectory_weight * magnitude_trajectory_loss
            )
        return tuple(losses) if len(losses) > 1 else losses[0]
