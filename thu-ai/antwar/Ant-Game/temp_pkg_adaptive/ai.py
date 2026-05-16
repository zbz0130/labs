from __future__ import annotations

from dataclasses import dataclass
from math import inf
import os
from pathlib import Path

import numpy as np

try:
    from common import BaseAgent
except ModuleNotFoundError as exc:
    if exc.name != "common":
        raise
    from AI.common import BaseAgent

from SDK.alphazero import PolicyValueInference, PolicyValueNet, PriorGuidedMCTS, SearchConfig, infer_observation_dim
from SDK.backend import BackendState
from SDK.backend.model import Operation
from SDK.utils.actions import ActionBundle
from SDK.utils.constants import (
    BASIC_INCOME,
    MAX_ACTIONS,
    OperationType,
    PLAYER_BASES,
    SUPER_WEAPON_STATS,
    SuperWeaponType,
    TowerType,
)
from SDK.utils.geometry import hex_distance
from SDK.utils.turns import DecisionContext


CONTROL_TOWERS = {TowerType.ICE, TowerType.BEWITCH, TowerType.PULSE}
AOE_TOWERS = {TowerType.MORTAR, TowerType.MORTAR_PLUS, TowerType.PULSE, TowerType.MISSILE}
FAST_TOWERS = {TowerType.QUICK, TowerType.QUICK_PLUS, TowerType.DOUBLE, TowerType.SNIPER}
HEAVY_TOWERS = {TowerType.HEAVY, TowerType.HEAVY_PLUS}
PRODUCER_TOWERS = {
    TowerType.PRODUCER,
    TowerType.PRODUCER_FAST,
    TowerType.PRODUCER_SIEGE,
    TowerType.PRODUCER_MEDIC,
}
OPENING_BUILD_SEQUENCE = {
    0: ((6, 9), (5, 9), (5, 11), (8, 11), (7, 10), (8, 7)),
    1: ((12, 9), (13, 9), (13, 7), (10, 7), (11, 8), (10, 11)),
}


@dataclass(slots=True)
class TacticalProfile:
    round_index: int
    my_base_hp: int
    enemy_base_hp: int
    my_coins: int
    enemy_coins: int
    my_tower_count: int
    enemy_tower_count: int
    my_frontline: int
    enemy_frontline: int
    my_push: float
    enemy_threat: float
    ally_contest_mass: float
    enemy_contest_mass: float
    enemy_tower_guard: float
    my_producers: int
    enemy_producers: int
    my_combat: int
    enemy_combat: int
    safe_buffer: int

    @property
    def emergency(self) -> bool:
        return self.my_base_hp <= 18 or self.enemy_frontline <= 5 or self.enemy_threat >= 24.0

    @property
    def attack_window(self) -> float:
        return self.my_push - 0.55 * self.enemy_tower_guard - 0.35 * self.enemy_threat

    @property
    def early(self) -> bool:
        return self.round_index < 90

    @property
    def late(self) -> bool:
        return self.round_index >= 240 or self.my_base_hp <= 20 or self.enemy_base_hp <= 20


class AdaptiveAgent(BaseAgent):
    """Adaptive heuristic agent with lightweight forward simulation.

    The SDK already owns legality checks and round simulation, so this agent only
    needs to rank candidate bundles. The strategy blends:
    1. local tactical bonuses for build/upgrade/weapon choices
    2. one-round lookahead against a predicted opponent reply
    3. phase-aware pressure / economy preferences
    """

    search_width = 16
    enemy_width = 8
    followup_width = 5
    deep_width = 4
    mcts_root_width = 8

    def __init__(
        self,
        seed: int | None = None,
        max_actions: int = MAX_ACTIONS,
        *,
        mcts_enabled: bool = True,
        mcts_model_path: str | os.PathLike[str] | None = None,
    ) -> None:
        super().__init__(seed=seed, max_actions=max_actions)
        self.mcts_enabled = mcts_enabled
        self._mcts_model_path = mcts_model_path
        self.mcts_model = self._load_mcts_model(mcts_model_path) if mcts_enabled else None

    def list_bundles(self, state: BackendState, player: int) -> list[ActionBundle]:
        return self.catalog.build(state, player, context=DecisionContext.for_player(player), rerank=False)

    def choose_bundle(
        self,
        state: BackendState,
        player: int,
        bundles: list[ActionBundle] | None = None,
    ) -> ActionBundle:
        bundles = bundles or self.list_bundles(state, player)
        if not bundles:
            return ActionBundle(name="hold", score=0.0, tags=("noop",))
        if len(bundles) == 1:
            return bundles[0]

        context = DecisionContext.for_player(player)
        profile = self._profile(state, player)
        baseline_value = self._state_value(state, player, profile)
        search_width, _, _, deep_width = self._search_budget(profile)
        shortlist = self._shortlist(bundles, state, player, profile, width=search_width)

        scored: list[tuple[float, ActionBundle, BackendState | None]] = []
        for bundle in shortlist:
            score, settled_state = self._project_bundle(
                state,
                player,
                bundle,
                context=context,
                baseline_value=baseline_value,
                profile=profile,
            )
            scored.append((score, bundle, settled_state))

        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_bundle, _ = scored[0]

        if self.mcts_enabled and self._should_use_mcts(profile, scored):
            mcts_bundle = self._choose_with_mcts(state, player, profile, scored)
            if mcts_bundle is not None:
                return mcts_bundle

        for score, bundle, settled_state in scored[:deep_width]:
            if settled_state is None or settled_state.terminal:
                refined = score
            else:
                refined = score + self._followup_hint(settled_state, player)
            if refined > best_score:
                best_score = refined
                best_bundle = bundle

        return best_bundle

    def _shortlist(
        self,
        bundles: list[ActionBundle],
        state: BackendState,
        player: int,
        profile: TacticalProfile,
        *,
        width: int,
    ) -> list[ActionBundle]:
        picked: dict[tuple[tuple[int, int, int], ...], ActionBundle] = {}

        def add(bundle: ActionBundle | None) -> None:
            if bundle is None:
                return
            key = tuple((int(op.op_type), op.arg0, op.arg1) for op in bundle.operations)
            picked.setdefault(key, bundle)

        add(bundles[0])
        for bundle in bundles[1 : min(len(bundles), width + 4)]:
            add(bundle)
        add(self._opening_bundle(state, player, bundles, profile))

        for tag in ("build", "upgrade", "base", "weapon", "combo"):
            candidate = max((bundle for bundle in bundles if tag in bundle.tags), key=lambda item: item.score, default=None)
            add(candidate)

        if profile.round_index >= 90 or profile.safe_buffer < -12:
            sell_candidate = max((bundle for bundle in bundles if "sell" in bundle.tags), key=lambda item: item.score, default=None)
            add(sell_candidate)

        if profile.emergency:
            for bundle in bundles:
                if any(tag in bundle.tags for tag in ("build", "upgrade", "weapon")):
                    add(bundle)
                if len(picked) >= width + 8:
                    break

        ordered = sorted(picked.values(), key=lambda item: item.score, reverse=True)
        return ordered[:width]

    def _search_budget(self, profile: TacticalProfile) -> tuple[int, int, int, int]:
        search = self.search_width
        enemy = self.enemy_width
        followup = self.followup_width
        deep = self.deep_width

        if profile.early and profile.my_tower_count <= 2 and not profile.emergency:
            search -= 4
            enemy -= 2
            followup -= 1
            deep -= 1
        if profile.attack_window >= 6.0 or profile.enemy_frontline <= 8:
            search += 3
            enemy += 1
            followup += 1
        if profile.emergency:
            search += 5
            enemy += 2
            followup += 1
            deep += 1
        if profile.late:
            search += 2
            enemy += 1

        return max(search, 8), max(enemy, 4), max(followup, 3), max(deep, 2)

    def _should_use_mcts(
        self,
        profile: TacticalProfile,
        scored: list[tuple[float, ActionBundle, BackendState | None]],
    ) -> bool:
        if len(scored) <= 1:
            return False
        if profile.emergency or profile.late:
            return True
        if profile.attack_window >= 6.0:
            return True
        top_tags = {tag for _, bundle, _ in scored[:4] for tag in bundle.tags}
        if "weapon" in top_tags or "combo" in top_tags:
            return True
        return scored[0][0] - scored[min(2, len(scored) - 1)][0] <= 5.0

    def _choose_with_mcts(
        self,
        state: BackendState,
        player: int,
        profile: TacticalProfile,
        scored: list[tuple[float, ActionBundle, BackendState | None]],
    ) -> ActionBundle | None:
        iterations, max_depth, root_limit, child_limit = self._mcts_budget(profile)
        bundles = [bundle for _, bundle, _ in scored[:root_limit]]
        if len(bundles) <= 1:
            return None
        search = AdaptiveGuidedMCTS(
            agent=self,
            model=self.mcts_model,
            search_config=SearchConfig(
                iterations=iterations,
                max_depth=max_depth,
                c_puct=1.2,
                root_action_limit=root_limit,
                child_action_limit=child_limit,
                prior_mix=0.65,
                value_mix=0.75,
                value_scale=350.0,
                seed=self.rng.randrange(1 << 30),
            ),
        )
        result = search.search(
            state=state,
            player=player,
            bundles=bundles,
            context=DecisionContext.for_player(player),
            temperature=1e-6,
            add_root_noise=False,
        )
        if not result.bundle.operations and scored[0][0] > 3.0:
            return scored[0][1]
        return result.bundle

    def _mcts_budget(self, profile: TacticalProfile) -> tuple[int, int, int, int]:
        iterations = 10
        max_depth = 2
        root_limit = self.mcts_root_width
        child_limit = 5
        if profile.attack_window >= 6.0:
            iterations += 4
            child_limit += 1
        if profile.emergency:
            iterations += 8
            max_depth += 1
            root_limit += 2
            child_limit += 1
        if profile.late:
            iterations += 4
            max_depth += 1
        return iterations, max_depth, min(root_limit, 12), min(child_limit, 7)

    def _opening_bundle(
        self,
        state: BackendState,
        player: int,
        bundles: list[ActionBundle],
        profile: TacticalProfile,
    ) -> ActionBundle | None:
        if not profile.early or state.round_index > 48:
            return None
        sequence = OPENING_BUILD_SEQUENCE[player]
        tower_count = state.tower_count(player)
        if tower_count >= len(sequence):
            return None
        if state.coins[player] < state.build_tower_cost(tower_count):
            return None
        target = sequence[tower_count]
        matching = [
            bundle
            for bundle in bundles
            if any(
                op.op_type == OperationType.BUILD_TOWER and (op.arg0, op.arg1) == target
                for op in bundle.operations
            )
        ]
        if not matching:
            return None
        return max(matching, key=lambda item: item.score)

    def _project_bundle(
        self,
        state: BackendState,
        player: int,
        bundle: ActionBundle,
        *,
        context: DecisionContext,
        baseline_value: float,
        profile: TacticalProfile,
    ) -> tuple[float, BackendState | None]:
        trial = state.clone()
        invalid = trial.apply_operation_list(player, bundle.operations)
        if invalid:
            return -inf, None

        local = 0.7 * bundle.score + self._bundle_local_bias(state, trial, player, bundle, profile)
        settled = trial.clone()

        if not settled.terminal and not context.settles_after_action:
            enemy = 1 - player
            enemy_reply = self._predict_enemy_reply(settled, enemy)
            if enemy_reply.operations:
                settled.apply_operation_list(enemy, enemy_reply.operations)

        if not settled.terminal:
            settled.advance_round()

        future_profile = self._profile(settled, player)
        future_value = self._state_value(settled, player, future_profile)
        delta = future_value - baseline_value
        score = local + 0.6 * delta
        return score, settled

    def _predict_enemy_reply(self, state: BackendState, enemy: int) -> ActionBundle:
        bundles = self.catalog.build(state, enemy, context=DecisionContext.for_player(enemy), rerank=False)
        if not bundles:
            return ActionBundle(name="hold", score=0.0, tags=("noop",))

        profile = self._profile(state, enemy)
        baseline = self._state_value(state, enemy, profile)
        _, enemy_width, _, _ = self._search_budget(profile)
        shortlist = self._shortlist(bundles, state, enemy, profile, width=enemy_width)
        best_bundle = shortlist[0]
        best_score = -inf

        for bundle in shortlist:
            trial = state.clone()
            invalid = trial.apply_operation_list(enemy, bundle.operations)
            if invalid:
                continue
            if not trial.terminal:
                trial.advance_round()
            future_profile = self._profile(trial, enemy)
            future_value = self._state_value(trial, enemy, future_profile)
            score = 0.65 * bundle.score + self._bundle_local_bias(state, trial, enemy, bundle, profile) + 0.55 * (future_value - baseline)
            if score > best_score:
                best_score = score
                best_bundle = bundle
        return best_bundle

    def _followup_hint(self, state: BackendState, player: int) -> float:
        # Only player 0 acts first after the next settle, so player 1 should not
        # over-trust a self-follow-up that happens after the opponent has moved.
        if player != 0:
            return 0.0

        bundles = self.catalog.build(state, player, context=DecisionContext.for_player(player), rerank=False)
        if not bundles:
            return 0.0

        profile = self._profile(state, player)
        baseline = self._state_value(state, player, profile)
        _, _, followup_width, _ = self._search_budget(profile)
        best = 0.0
        for bundle in self._shortlist(bundles, state, player, profile, width=followup_width):
            trial = state.clone()
            invalid = trial.apply_operation_list(player, bundle.operations)
            if invalid:
                continue
            future_profile = self._profile(trial, player)
            value = self._state_value(trial, player, future_profile) - baseline
            hint = 0.25 * bundle.score + self._bundle_local_bias(state, trial, player, bundle, profile) + 0.35 * value
            best = max(best, hint)
        return 0.2 * best

    def _state_value(self, state: BackendState, player: int, profile: TacticalProfile | None = None) -> float:
        if profile is None:
            profile = self._profile(state, player)

        value = self.feature_extractor.evaluate(state, player, context=DecisionContext.for_player(player))
        value += profile.my_push * 2.0
        value -= profile.enemy_threat * 3.5
        value += profile.ally_contest_mass * 1.2
        value -= profile.enemy_contest_mass * 1.4
        value += (profile.my_combat - profile.enemy_combat) * 2.4
        value += min(profile.safe_buffer, 60) * 0.1
        value -= max(-profile.safe_buffer, 0) * 0.25
        value += max(0, 12 - profile.my_frontline) * 1.0
        value -= max(0, 12 - profile.enemy_frontline) * 1.8
        value += profile.my_producers * max(0.0, 4.5 - profile.enemy_threat * 0.12)
        value -= profile.enemy_producers * 1.8
        value -= profile.enemy_tower_guard * 0.4

        if profile.emergency:
            value -= max(0, 8 - profile.enemy_frontline) * 6.0
            value -= max(0, 20 - profile.my_base_hp) * 3.0
        if profile.attack_window > 0:
            value += profile.attack_window * 0.9
        if profile.enemy_base_hp <= 12:
            value += (12 - profile.enemy_base_hp) * 4.0
        if profile.my_base_hp <= 12:
            value -= (12 - profile.my_base_hp) * 5.5

        for effect in state.active_effects:
            owner_sign = 1.0 if effect.player == player else -1.0
            remaining_ratio = effect.remaining_turns / max(SUPER_WEAPON_STATS[effect.weapon_type].duration, 1)
            if effect.weapon_type == SuperWeaponType.LIGHTNING_STORM:
                value += owner_sign * 6.0 * remaining_ratio
            elif effect.weapon_type == SuperWeaponType.EMP_BLASTER:
                value += owner_sign * 5.0 * remaining_ratio
            elif effect.weapon_type == SuperWeaponType.DEFLECTOR:
                value += owner_sign * 3.5 * remaining_ratio
            elif effect.weapon_type == SuperWeaponType.EMERGENCY_EVASION:
                value += owner_sign * 2.0 * remaining_ratio

        return value

    def _bundle_local_bias(
        self,
        before: BackendState,
        after: BackendState,
        player: int,
        bundle: ActionBundle,
        profile: TacticalProfile,
    ) -> float:
        if not bundle.operations:
            return self._hold_bias(before, player, profile)

        bonus = 0.0
        for operation in bundle.operations:
            bonus += self._operation_bias(before, after, player, operation, profile)

        if len(bundle.operations) >= 2:
            bonus += 2.0
            types = {op.op_type for op in bundle.operations}
            if OperationType.DOWNGRADE_TOWER in types and any(
                op_type in types
                for op_type in (
                    OperationType.BUILD_TOWER,
                    OperationType.UPGRADE_TOWER,
                    OperationType.UPGRADE_GENERATION_SPEED,
                    OperationType.UPGRADE_GENERATED_ANT,
                )
            ):
                bonus += 3.5

        if "sell" in bundle.tags and len(bundle.operations) == 1:
            bonus -= 5.0

        reserve = after.coins[player] - after.safe_coin_threshold(player)
        if reserve >= 0:
            bonus += min(reserve, 40) * 0.08
        else:
            bonus += reserve * 0.45
        if profile.emergency and reserve < 0:
            bonus += reserve * 0.25

        bonus += self._inventory_bonus(after, player, profile)
        return bonus

    def _hold_bias(self, state: BackendState, player: int, profile: TacticalProfile) -> float:
        if profile.emergency:
            return -8.0
        bonus = 0.0
        soon_affordable = min(
            (
                max(stats.cost - state.coins[player], 0)
                for stats in SUPER_WEAPON_STATS.values()
                if stats.cost > state.coins[player]
            ),
            default=999,
        )
        if soon_affordable <= BASIC_INCOME + 6:
            bonus += 3.0
        if profile.safe_buffer >= 20 and profile.my_tower_count >= 5:
            bonus += 1.5
        if state.coins[player] < state.build_tower_cost(state.tower_count(player)) and profile.enemy_frontline > 8:
            bonus += 2.0
        if profile.early and state.coins[player] >= state.build_tower_cost(state.tower_count(player)) and profile.my_tower_count < 4:
            bonus -= 4.0
        return bonus

    def _operation_bias(
        self,
        before: BackendState,
        after: BackendState,
        player: int,
        operation: Operation,
        profile: TacticalProfile,
    ) -> float:
        enemy = 1 - player
        enemy_base = PLAYER_BASES[enemy]
        my_base = PLAYER_BASES[player]

        if operation.op_type == OperationType.BUILD_TOWER:
            x, y = operation.arg0, operation.arg1
            pressure = self._cell_pressure(before, player, x, y)
            bonus = before.slot_priority(player, x, y) * 0.45 + pressure * 1.3
            bonus += self._opening_build_bonus(before, player, x, y, profile)
            if profile.early and before.tower_count(player) < 3:
                bonus += 7.0
            if profile.enemy_frontline <= 7:
                bonus += 6.0
            if hex_distance(x, y, *enemy_base) < hex_distance(*my_base, *enemy_base):
                bonus += 1.2
            if profile.my_tower_count >= 6 and profile.safe_buffer < 10:
                bonus -= 3.0
            return bonus

        if operation.op_type == OperationType.UPGRADE_TOWER:
            tower = before.tower_by_id(operation.arg0)
            if tower is None:
                return -5.0
            target = TowerType(operation.arg1)
            local_density = self._cell_pressure(before, player, tower.x, tower.y)
            heal_value = max(tower.max_hp - tower.hp, 0) * 0.45
            dist_to_enemy = hex_distance(tower.x, tower.y, *enemy_base)

            if target in CONTROL_TOWERS:
                bonus = 7.0 + local_density * 1.4 + heal_value
                if profile.enemy_frontline <= 8:
                    bonus += 5.5
                return bonus
            if target in HEAVY_TOWERS:
                bonus = 5.0 + local_density * 1.1 + heal_value
                if profile.emergency:
                    bonus += 4.0
                return bonus
            if target in AOE_TOWERS:
                bonus = 4.5 + local_density * 1.0 + heal_value
                bonus += max(0.0, 14 - dist_to_enemy) * 0.15
                return bonus
            if target in FAST_TOWERS:
                bonus = 4.5 + local_density * 0.85 + heal_value
                if target == TowerType.SNIPER:
                    bonus += max(0.0, 16 - dist_to_enemy) * 0.3
                return bonus
            if target in PRODUCER_TOWERS:
                safe_scale = max(0.0, profile.enemy_frontline - 7) * 0.7 + max(profile.safe_buffer, 0) * 0.04
                bonus = 2.0 + safe_scale + heal_value
                if target == TowerType.PRODUCER_FAST:
                    bonus += 3.0 + (5.0 if profile.early else 0.0)
                elif target == TowerType.PRODUCER_SIEGE:
                    bonus += 4.5 + max(0.0, 12 - profile.my_frontline) * 0.5
                elif target == TowerType.PRODUCER_MEDIC:
                    bonus += 4.0 + profile.ally_contest_mass * 0.35 + profile.my_combat * 1.0
                if profile.enemy_frontline <= 8:
                    bonus -= 7.5
                return bonus
            return heal_value

        if operation.op_type == OperationType.DOWNGRADE_TOWER:
            tower = before.tower_by_id(operation.arg0)
            refund = before.operation_income(player, operation)
            bonus = -10.0 + refund * 0.06
            if profile.round_index < 90:
                bonus -= 12.0
            if before.tower_count(player) <= 4:
                bonus -= 8.0
            if tower is not None and tower.hp <= max(1, tower.max_hp // 2):
                bonus += 3.0
            if tower is not None:
                if tower.tower_type == TowerType.BASIC and tower.hp == tower.max_hp:
                    bonus -= 10.0
                bonus -= before.slot_priority(player, tower.x, tower.y) * 0.18
            if profile.emergency:
                bonus -= 5.0
            return bonus

        if operation.op_type == OperationType.UPGRADE_GENERATION_SPEED:
            bonus = 14.0
            if profile.early:
                bonus += 6.0
            if before.tower_count(player) < 3:
                bonus -= 3.5
            if profile.enemy_frontline <= 7:
                bonus -= 8.0
            if profile.safe_buffer < 20:
                bonus -= 2.5
            return bonus

        if operation.op_type == OperationType.UPGRADE_GENERATED_ANT:
            bonus = 10.0
            if not profile.early:
                bonus += 4.0
            if before.tower_count(player) < 2:
                bonus -= 2.5
            if profile.attack_window > 0:
                bonus += 5.0
            if profile.enemy_frontline <= 7:
                bonus -= 5.0
            return bonus

        if operation.op_type == OperationType.USE_LIGHTNING_STORM:
            bonus = 4.0
            if profile.enemy_threat >= 14.0:
                bonus += 4.5
            if profile.attack_window > 0:
                bonus += 3.0
            if profile.early and profile.enemy_frontline > 9:
                bonus -= 5.0
            return bonus

        if operation.op_type == OperationType.USE_EMP_BLASTER:
            bonus = 5.5 + profile.enemy_tower_count * 0.7
            bonus += max(0.0, 12 - profile.my_frontline) * 0.5
            if profile.attack_window < 2.0:
                bonus -= 4.0
            return bonus

        if operation.op_type == OperationType.USE_DEFLECTOR:
            bonus = 4.0 + profile.ally_contest_mass * 0.45 + profile.my_combat * 1.0
            if profile.ally_contest_mass < 3.0:
                bonus -= 3.0
            return bonus

        if operation.op_type == OperationType.USE_EMERGENCY_EVASION:
            bonus = 4.5 + profile.ally_contest_mass * 0.4
            if profile.enemy_frontline <= 6:
                bonus += 3.5
            if profile.ally_contest_mass < 2.5:
                bonus -= 3.0
            return bonus

        return 0.0

    def _inventory_bonus(self, state: BackendState, player: int, profile: TacticalProfile) -> float:
        control = 0
        aoe = 0
        producers = 0
        fast = 0
        durable = 0
        for tower in state.towers_of(player):
            tower_type = tower.tower_type
            if tower_type in CONTROL_TOWERS:
                control += 1
            if tower_type in AOE_TOWERS:
                aoe += 1
            if tower_type in PRODUCER_TOWERS:
                producers += 1
            if tower_type in FAST_TOWERS:
                fast += 1
            if tower_type in HEAVY_TOWERS:
                durable += 1

        bonus = 0.0
        if profile.enemy_frontline <= 8:
            bonus += control * 1.6 + aoe * 1.3 + durable * 1.0
        else:
            bonus += producers * 1.6 + fast * 0.9
        if profile.early:
            bonus += min(producers, 1) * max(0.0, 3.5 - profile.enemy_threat * 0.08)
        if profile.late:
            bonus += control * 0.8 + aoe * 0.7
        return bonus

    def _profile(self, state: BackendState, player: int) -> TacticalProfile:
        enemy = 1 - player
        my_base = PLAYER_BASES[player]
        enemy_base = PLAYER_BASES[enemy]
        my_ants = [ant for ant in state.ants_of(player) if ant.is_alive()]
        enemy_ants = [ant for ant in state.ants_of(enemy) if ant.is_alive()]
        my_towers = state.towers_of(player)
        enemy_towers = state.towers_of(enemy)

        my_push = 0.0
        ally_contest_mass = 0.0
        my_combat = 0
        for ant in my_ants:
            distance = hex_distance(ant.x, ant.y, *enemy_base)
            hp_ratio = ant.hp / max(ant.max_hp, 1)
            power = (1.0 + ant.level * 0.35 + (1.2 if int(ant.kind) == 1 else 0.0)) * hp_ratio
            my_push += max(0.0, 15.0 - distance) * power
            if distance <= 8:
                ally_contest_mass += power
            if int(ant.kind) == 1:
                my_combat += 1

        enemy_threat = 0.0
        enemy_contest_mass = 0.0
        enemy_combat = 0
        for ant in enemy_ants:
            distance = hex_distance(ant.x, ant.y, *my_base)
            hp_ratio = ant.hp / max(ant.max_hp, 1)
            power = (1.0 + ant.level * 0.4 + (1.4 if int(ant.kind) == 1 else 0.0)) * hp_ratio
            enemy_threat += max(0.0, 15.0 - distance) * power
            if distance <= 8:
                enemy_contest_mass += power
            if int(ant.kind) == 1:
                enemy_combat += 1

        enemy_tower_guard = 0.0
        for tower in enemy_towers:
            dist = hex_distance(tower.x, tower.y, *enemy_base)
            enemy_tower_guard += max(0.0, 12.0 - dist) * (1.2 + tower.level * 0.8)

        my_producers = sum(1 for tower in my_towers if tower.is_producer)
        enemy_producers = sum(1 for tower in enemy_towers if tower.is_producer)

        return TacticalProfile(
            round_index=state.round_index,
            my_base_hp=state.bases[player].hp,
            enemy_base_hp=state.bases[enemy].hp,
            my_coins=state.coins[player],
            enemy_coins=state.coins[enemy],
            my_tower_count=len(my_towers),
            enemy_tower_count=len(enemy_towers),
            my_frontline=state.frontline_distance(player),
            enemy_frontline=state.nearest_ant_distance(player),
            my_push=my_push,
            enemy_threat=enemy_threat,
            ally_contest_mass=ally_contest_mass,
            enemy_contest_mass=enemy_contest_mass,
            enemy_tower_guard=enemy_tower_guard,
            my_producers=my_producers,
            enemy_producers=enemy_producers,
            my_combat=my_combat,
            enemy_combat=enemy_combat,
            safe_buffer=state.coins[player] - state.safe_coin_threshold(player),
        )

    def _opening_build_bonus(
        self,
        state: BackendState,
        player: int,
        x: int,
        y: int,
        profile: TacticalProfile,
    ) -> float:
        if not profile.early or state.round_index > 56:
            return 0.0
        sequence = OPENING_BUILD_SEQUENCE[player]
        tower_count = state.tower_count(player)
        if tower_count >= len(sequence):
            return 0.0
        target = sequence[tower_count]
        if (x, y) == target:
            return 10.0 - tower_count * 1.3
        if tower_count <= 1 and (x, y) in sequence[:3]:
            return 4.0
        if tower_count <= 2 and (x, y) not in sequence[:4]:
            return -2.5
        return 0.0

    def _candidate_model_paths(self, override: str | os.PathLike[str] | None) -> list[Path]:
        candidates: list[Path] = []
        if override is not None:
            candidates.append(Path(override))
            return candidates
        env_path = os.getenv("AGENT_TRADITION_MCTS_MODEL")
        if env_path:
            candidates.append(Path(env_path))
        module_root = Path(__file__).resolve().parent
        repo_root = module_root.parent
        candidates.extend(
            [
                module_root / "ai_mcts_model.npz",
                repo_root / "checkpoints" / "ai_mcts_latest.npz",
                repo_root / "SDK" / "checkpoints" / "ai_mcts_latest.npz",
            ]
        )
        return candidates

    def _load_mcts_model(self, model_path: str | os.PathLike[str] | None) -> PolicyValueNet | None:
        expected_obs_dim = infer_observation_dim(self.feature_extractor, self.catalog.max_actions)
        for candidate in self._candidate_model_paths(model_path):
            if not candidate.exists():
                continue
            try:
                model = PolicyValueNet.from_checkpoint(candidate)
            except (OSError, ValueError, KeyError):
                continue
            if model.action_dim != self.catalog.max_actions:
                continue
            if model.obs_dim != expected_obs_dim:
                continue
            return model
        return None

    def _cell_pressure(self, state: BackendState, player: int, x: int, y: int) -> float:
        pressure = 0.0
        for ant in state.ants_of(1 - player):
            if not ant.is_alive():
                continue
            distance = hex_distance(x, y, ant.x, ant.y)
            if distance > 7:
                continue
            hp_ratio = ant.hp / max(ant.max_hp, 1)
            weight = 1.0 + ant.level * 0.35 + (1.3 if int(ant.kind) == 1 else 0.0)
            pressure += max(0.0, 7.5 - distance) * weight * hp_ratio
        return pressure


class AdaptiveGuidedMCTS(PriorGuidedMCTS):
    def __init__(
        self,
        agent: AdaptiveAgent,
        model: PolicyValueNet | None = None,
        search_config: SearchConfig | None = None,
    ) -> None:
        super().__init__(
            model=model,
            search_config=search_config,
            feature_extractor=agent.feature_extractor,
            action_catalog=agent.catalog,
        )
        self.agent = agent

    def _heuristic_value(
        self,
        state: BackendState,
        player: int,
        context: DecisionContext | None = None,
    ) -> float:
        if state.terminal:
            if state.winner is None:
                return 0.0
            return 1.0 if state.winner == player else -1.0
        profile = self.agent._profile(state, player)
        raw = self.agent._state_value(state, player, profile)
        return float(np.tanh(raw / self.search_config.value_scale))

    def _blend_policy_value(
        self,
        state: BackendState,
        player: int,
        context: DecisionContext,
        bundles: list[ActionBundle],
    ):
        action_mask = self.action_catalog.action_mask(bundles).astype(np.float32)
        observation = self.feature_extractor.encode_observation(state, player, action_mask, context=context)
        flat = self.feature_extractor.flatten_observation(observation)
        heuristic_priors = self._adaptive_bundle_policy(state, player, bundles)
        heuristic_value = self._heuristic_value(state, player, context=context)
        if self.model is None:
            blended_priors = heuristic_priors
            blended_value = heuristic_value
        else:
            model_priors, model_value = self.model.predict(flat, action_mask)
            mixed_policy = self.search_config.prior_mix * model_priors[: len(bundles)]
            mixed_policy += (1.0 - self.search_config.prior_mix) * heuristic_priors
            blended_priors = self._normalize_policy(mixed_policy)
            blended_value = float(
                self.search_config.value_mix * model_value
                + (1.0 - self.search_config.value_mix) * heuristic_value
            )
        full_priors = np.zeros(self.action_dim, dtype=np.float32)
        full_priors[: len(bundles)] = blended_priors
        return PolicyValueInference(
            priors=full_priors,
            value=float(blended_value),
            observation=flat,
            mask=action_mask,
        )

    def _adaptive_bundle_policy(
        self,
        state: BackendState,
        player: int,
        bundles: list[ActionBundle],
    ) -> np.ndarray:
        if not bundles:
            return np.zeros(0, dtype=np.float32)
        profile = self.agent._profile(state, player)
        scores: list[float] = []
        for bundle in bundles:
            if not bundle.operations:
                score = bundle.score + self.agent._hold_bias(state, player, profile)
            else:
                trial = state.clone()
                invalid = trial.apply_operation_list(player, bundle.operations)
                if invalid:
                    score = -1e6
                else:
                    score = bundle.score + 0.35 * self.agent._bundle_local_bias(state, trial, player, bundle, profile)
            scores.append(score)
        logits = np.asarray(scores, dtype=np.float32)
        logits = (logits - np.max(logits)) / 6.0
        exp = np.exp(logits).astype(np.float32, copy=False)
        return self._normalize_policy(exp)

    @staticmethod
    def _normalize_policy(policy: np.ndarray) -> np.ndarray:
        total = float(np.sum(policy))
        if total <= 0.0:
            fallback = np.zeros_like(policy, dtype=np.float32)
            if fallback.size:
                fallback[0] = 1.0
            return fallback
        return (policy / total).astype(np.float32, copy=False)


class AI(AdaptiveAgent):
    pass
