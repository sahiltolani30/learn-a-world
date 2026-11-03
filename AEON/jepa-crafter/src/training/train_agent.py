"""Main training and orchestration loop for JEPA-Crafter agent."""
import os
import sys
import time
from typing import Optional, Dict, Any
import numpy as np
import torch

from src.utils.config import Config, load_config
from src.utils.seeding import seed_everything, get_rng_states
from src.utils.device import get_device
from src.utils.logging import MetricLogger, save_json
from src.env.crafter_adapter import CrafterAdapter
from src.env.vector_adapter import VectorCrafterAdapter
from src.env.evaluator_tap import EvaluatorTap
from src.env.opaque_actions import OpaqueActionMapping
from src.data.replay_buffer import ReplayBuffer
from src.data.sequence_sampler import SequenceSampler
from src.data.checkpoint_store import CheckpointStore
from src.models.jepa_world_model import JepaWorldModel
from src.learning.learner import JepaLearner
from src.learning.optimizer import build_optimizer_and_scheduler
from src.learning.update_scheduler import UpdateScheduler
from src.learning.feature_diagnostics import compute_feature_diagnostics
from src.exploration.novelty import NoveltyEstimator
from src.exploration.disagreement import DisagreementEstimator, LearningProgressEstimator
from src.exploration.action_selector import ActionSelector
from src.exploration.role_definitions import ROLE_WEIGHTS
from src.planning.candidate_scorer import CandidateScorer
from src.planning.cem_planner import CemPlanner
from src.planning.planner_diagnostics import PlannerDiagnostics
from src.training.collect import BatchedExperienceCollector
from src.evaluation.achievement_recorder import AchievementRecorder
from src.evaluation.heldout_prediction import HeldoutPredictorEvaluator
from src.evaluation.action_conditioning_tests import ActionConditioningTester
from src.evaluation.report import ReportGenerator


def run_training(config: Config, max_steps: Optional[int] = None) -> Dict[str, Any]:
    """Run full agent training and evaluation according to config."""
    # 1. Output directory and seeds
    run_dir = os.path.join(config.project.output_dir, config.project.name)
    os.makedirs(run_dir, exist_ok=True)
    seed_everything(config.project.seed)
    device = get_device(config.project.device)

    target_env_steps = max_steps if max_steps is not None else config.environment.max_environment_steps

    # 2. Logging and checkpointing
    metric_logger = MetricLogger(run_dir, prefix="metrics")
    checkpoint_store = CheckpointStore(os.path.join(run_dir, "checkpoints"))
    report_gen = ReportGenerator(run_dir)
    report_gen.write_environment_manifest()

    # 3. Environment adapter with anonymous action mapping
    evaluator_tap = EvaluatorTap()
    action_mapping = OpaqueActionMapping(num_actions=config.environment.action_count, seed=config.project.seed)
    num_envs = getattr(config.collection, "num_env_workers", 1)

    if num_envs > 1:
        vec_adapter = VectorCrafterAdapter(
            num_envs=num_envs,
            env_id=config.environment.env_id,
            reward_mode=config.environment.reward_mode,
            seed_start=config.collection.train_seed_start,
            opaque_mapping=action_mapping,
            evaluator_tap=evaluator_tap,
        )
        adapter = None
    else:
        vec_adapter = None
        adapter = CrafterAdapter(
            env_id=config.environment.env_id,
            reward_mode=config.environment.reward_mode,
            seed=config.project.seed,
            opaque_mapping=action_mapping,
            evaluator_tap=evaluator_tap,
        )

    # 4. Replay buffer and sequence sampler
    replay_buffer = ReplayBuffer(
        capacity=config.replay.capacity,
        obs_shape=tuple(config.environment.observation_shape),
    )
    sampler = SequenceSampler(
        replay_buffer=replay_buffer,
        context_length=config.replay.context_length,
        prediction_horizon=config.replay.prediction_horizon,
        recent_fraction=config.replay.recent_sequence_fraction,
    )

    # 5. World Model and Learner
    model = JepaWorldModel(
        latent_dim=config.model.latent_dim,
        action_dim=config.model.action_embedding_dim,
        context_dim=config.model.context_hidden_dim,
        num_actions=config.environment.action_count,
        num_predictor_members=config.model.predictor_members,
        num_transformer_layers=config.model.predictor_transformer_blocks,
        num_heads=config.model.transformer_heads,
        mlp_dim=config.model.transformer_mlp_dim,
    ).to(device)

    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        encoder_lr=config.optimization.encoder_lr,
        predictor_lr=config.optimization.predictor_lr,
        auxiliary_lr=config.optimization.auxiliary_lr,
        weight_decay=config.optimization.weight_decay,
        warmup_updates=config.optimization.warmup_updates,
    )

    learner = JepaLearner(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        precision=config.optimization.precision,
        grad_clip_norm=config.optimization.gradient_clip_norm,
        ema_momentum_start=config.model.ema_momentum_start,
        ema_momentum_end=config.model.ema_momentum_end,
        ema_momentum_ramp=config.model.ema_momentum_ramp_updates,
    )

    update_scheduler = UpdateScheduler(updates_per_env_step=config.optimization.updates_per_env_step)

    # 6. Exploration and Planning
    novelty_est = NoveltyEstimator(capacity=config.replay.capacity, latent_dim=config.model.latent_dim, clip_val=config.exploration.intrinsic_clip)
    disagreement_est = DisagreementEstimator(clip_val=config.exploration.intrinsic_clip)
    progress_est = LearningProgressEstimator(clip_val=config.exploration.intrinsic_clip)
    action_selector = ActionSelector(
        num_actions=config.environment.action_count,
        random_action_floor=config.exploration.random_action_floor,
        novelty_estimator=novelty_est,
        disagreement_estimator=disagreement_est,
        progress_estimator=progress_est,
    )

    candidate_scorer = CandidateScorer(
        novelty_estimator=novelty_est,
        discount=config.planner.discount,
        max_intrinsic_bonus_per_step=config.planner.max_intrinsic_bonus_per_step,
    )
    planner = CemPlanner(
        model=model,
        scorer=candidate_scorer,
        num_actions=config.environment.action_count,
        horizon=config.planner.horizon,
        population_size=config.planner.population,
        num_iterations=config.planner.iterations,
        num_elites=config.planner.elite_count,
        algorithm=config.planner.algorithm,
    )
    planner_diag = PlannerDiagnostics()

    # 7. Evaluators
    achievement_recorder = AchievementRecorder()
    heldout_evaluator = HeldoutPredictorEvaluator(model=model, horizons=config.model.prediction_horizons)
    action_tester = ActionConditioningTester(model=model)

    # 8. Main execution loop
    if vec_adapter is not None:
        batched_collector = BatchedExperienceCollector(
            vector_adapter=vec_adapter,
            replay_buffer=replay_buffer,
            action_selector=action_selector,
            model=model,
            planner=planner,
            device=device,
        )
        current_obs = None
    else:
        batched_collector = None
        current_obs = adapter.reset()
        replay_buffer.start_episode(current_obs, episode_id=1)

    episode_history_obs = [current_obs] if current_obs is not None else []
    episode_history_actions = [0]
    episode_step = 0
    current_role_idx = 0
    roles = config.population.roles

    mpc_unlocked = False
    validation_status = {"test_b": False, "test_c": False, "test_d": False}
    latest_heldout_metrics = {}
    latest_action_metrics = {}
    latest_diagnostics = {}

    print(f"Starting JEPA Crafter training run: {config.project.name}")
    print(f"Target steps: {target_env_steps}, Parallel Games: {num_envs}, Device: {device.type}, Parameters: {model.count_parameters()['total_trainable']:,}")

    env_step = 0
    while env_step < target_env_steps:
        # Determine mode: warmup vs explore vs mpc
        if replay_buffer.size < config.replay.min_sequences_before_training:
            mode = "warmup"
        elif not mpc_unlocked:
            mode = "explore"
        else:
            mode = "mpc"

        current_role = roles[current_role_idx % len(roles)]

        if batched_collector is not None:
            batch_info = batched_collector.step(mode=mode, role=current_role)
            step_inc = batch_info["num_steps"]
            env_step += step_inc

            if batch_info["dones"] > 0:
                current_role_idx += batch_info["dones"]
                summary = evaluator_tap.get_summary()
                for ach in summary["achievement_histories"][-batch_info["dones"]:]:
                    achievement_recorder.record_episode(ach)

            num_updates = update_scheduler.step(step_inc)
        else:
            env_step += 1
            latent = None
            context = None
            if mode != "warmup":
                model.eval()
                with torch.no_grad():
                    ctx_len = min(config.replay.context_length, len(episode_history_obs))
                    obs_slice = episode_history_obs[-ctx_len:]
                    act_slice = episode_history_actions[-ctx_len:]

                    obs_arr = np.zeros((config.replay.context_length, 64, 64, 3), dtype=np.uint8)
                    act_arr = np.zeros(config.replay.context_length, dtype=np.int64)
                    mask_arr = np.zeros(config.replay.context_length, dtype=bool)

                    start_idx = config.replay.context_length - ctx_len
                    for idx, (o, a) in enumerate(zip(obs_slice, act_slice)):
                        obs_arr[start_idx + idx] = o
                        act_arr[start_idx + idx] = a
                        mask_arr[start_idx + idx] = True

                    ch_obs = np.transpose(obs_arr, (0, 3, 1, 2))
                    t_obs = torch.from_numpy(ch_obs).float().div_(255.0).unsqueeze(0).to(device)
                    t_act = torch.from_numpy(act_arr).unsqueeze(0).to(device)
                    t_mask = torch.from_numpy(mask_arr).unsqueeze(0).to(device)

                    latent, context = model.encode_actual_history(t_obs, t_act, t_mask)

            t0_plan = time.time()
            action_token = action_selector.select_action(
                latent=latent,
                context=context,
                model=model,
                mode=mode,
                role=current_role,
                planner=planner if mpc_unlocked else None,
            )
            plan_latency = time.time() - t0_plan

            next_obs, reward, terminated, truncated = adapter.step(action_token)
            done = (terminated or truncated)
            episode_step += 1

            if mode == "mpc" and planner.last_plan_info:
                planner_diag.record_decision(
                    predicted_reward=planner.last_plan_info.get("predicted_first_reward", 0.0),
                    actual_reward=reward,
                    predicted_term=planner.last_plan_info.get("predicted_first_term", 0.0),
                    actual_term=float(done),
                    disagreement=planner.last_plan_info.get("predicted_first_disag", 0.0),
                    score=planner.last_plan_info.get("best_score", 0.0),
                    latency_sec=plan_latency,
                )

            replay_buffer.add(
                action=action_token,
                reward=reward,
                next_obs=next_obs,
                terminated=terminated,
                truncated=truncated,
            )

            episode_history_obs.append(next_obs)
            episode_history_actions.append(action_token)
            if len(episode_history_obs) > config.replay.context_length * 2:
                episode_history_obs.pop(0)
                episode_history_actions.pop(0)

            num_updates = update_scheduler.step(1)

            if done:
                summary = evaluator_tap.get_summary()
                if summary["achievement_histories"]:
                    latest_ach = summary["achievement_histories"][-1]
                    achievement_recorder.record_episode(latest_ach, total_reward=summary["mean_reward"], length=episode_step)

                current_role_idx += 1
                episode_step = 0
                current_obs = adapter.reset()
                replay_buffer.start_episode(current_obs)
                episode_history_obs = [current_obs]
                episode_history_actions = [0]

        # Learner updates
        if replay_buffer.size >= config.replay.min_sequences_before_training and num_updates > 0:
            for _ in range(num_updates):
                batch = sampler.sample_batch(batch_size=config.optimization.batch_size, device=device)
                train_metrics = learner.train_step(batch)

                # Periodic feature diagnostics (every 500 updates)
                if learner.update_count % 500 == 0:
                    with torch.no_grad():
                        z_diag, _ = model.online_encoder(batch["target_1"][:8])
                        diag_metrics = compute_feature_diagnostics(z_diag)
                        train_metrics.update(diag_metrics)
                        latest_diagnostics = diag_metrics
                    metric_logger.log(step=env_step, metrics=train_metrics)

        # Periodic Validation Gates & Evaluation Cadence
        eval_interval = min(10000, max(2500, target_env_steps // 10))
        if env_step % eval_interval == 0 and replay_buffer.size >= config.replay.min_sequences_before_training:
            print(f"[{env_step}/{target_env_steps}] Running validation gates...")
            with torch.no_grad():
                val_batch = sampler.sample_batch(batch_size=min(32, replay_buffer.size // 10), device=device)
                heldout_res = heldout_evaluator.evaluate_batch(val_batch)
                latest_heldout_metrics = heldout_res

            action_res = action_tester.run_paired_action_test(seeds=[1000, 1001, 1002])
            latest_action_metrics = action_res

            test_b_pass = heldout_res["one_step_beats_persistence"]
            test_c_pass = action_res["test_c_passed"]
            test_d_pass = (latest_diagnostics.get("collapse_warning", 0.0) == 0.0)

            validation_status = {"test_b": test_b_pass, "test_c": test_c_pass, "test_d": test_d_pass}
            print(f"Validation Status: Test B (Held-out beat persistence): {test_b_pass}, Test C (Action-conditioned): {test_c_pass}, Test D (No collapse): {test_d_pass}")

            if test_b_pass and test_c_pass and test_d_pass:
                mpc_unlocked = True
                print(">>> VALIDATION GATES PASSED: MPC UNLOCKED <<<")

            checkpoint_data = {
                "step": env_step,
                "learner_update_count": learner.update_count,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": learner.scaler.state_dict() if learner.scaler else {},
                "action_mapping": action_mapping.to_dict(),
                "replay_state": replay_buffer.get_state(),
                "rng_states": get_rng_states(),
                "validation_status": validation_status,
            }
            checkpoint_store.save(step=env_step, checkpoint_data=checkpoint_data)

    # 9. Final Benchmark Outputs & Artifact Generation
    print("Generating final research artifacts and reports...")
    if not latest_heldout_metrics and len(sampler.get_valid_anchor_indices()) >= 4:
        with torch.no_grad():
            val_batch = sampler.sample_batch(batch_size=min(16, len(sampler.get_valid_anchor_indices())), device=device)
            latest_heldout_metrics = heldout_evaluator.evaluate_batch(val_batch)
            z_diag, _ = model.online_encoder(val_batch["target_1"][:8])
            latest_diagnostics = compute_feature_diagnostics(z_diag)
        latest_action_metrics = action_tester.run_paired_action_test(seeds=[1000, 1001, 1002])

    ach_summary = achievement_recorder.get_summary()
    achievement_recorder.export_csv(
        os.path.join(run_dir, "achievement_rates.csv"),
        seed=config.project.seed,
        run_name=config.project.name,
    )
    save_json(latest_heldout_metrics, os.path.join(run_dir, "heldout_prediction.json"))
    save_json(latest_action_metrics, os.path.join(run_dir, "action_conditioning.json"))
    save_json(planner_diag.compute_imagination_gap(), os.path.join(run_dir, "planner_diagnostics.json"))

    final_report_path = report_gen.generate_final_report(
        config=config.to_dict(),
        heldout_metrics=latest_heldout_metrics,
        action_metrics=latest_action_metrics,
        achievement_summary=ach_summary,
        diagnostics=latest_diagnostics,
    )
    print(f"Final report generated at: {final_report_path}")

    return {
        "final_report_path": final_report_path,
        "achievement_summary": ach_summary,
        "validation_status": validation_status,
        "total_env_steps": target_env_steps,
    }
