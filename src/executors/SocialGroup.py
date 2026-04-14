"""
SocialGroup executor classifies person clusters into social relation types.
"""
import os
import sys
import time
import uuid
from itertools import combinations

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.Proximity.src.classes.StateManager import StateManager
from components.Proximity.src.classes.EventAccumulator import EventAccumulator
from components.Proximity.src.classes.GroupClassifier import GroupClassifier
from components.Proximity.src.models.PackageModel import PackageModel
from components.Proximity.src.utils.response import build_response_social_group


class SocialGroup(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))

        self.group_distance_m = self.request.get_param("ConfigGroupDistance")
        self.cohesion_window_s = self.request.get_param("ConfigCohesionWindow")
        self.trajectory_weight = self.request.get_param("ConfigTrajectoryWeight")
        self.family_age_gap = self.request.get_param("ConfigFamilyAgeGap")
        self.couple_mode = self.request.get_param("ConfigCoupleMode")
        self.min_friend_group_size = self.request.get_param("ConfigMinFriendGroupSize")
        self.confidence_threshold = self.request.get_param("ConfigConfidenceThreshold")
        self.relation_types_enabled = self.request.get_param("ConfigRelationTypes")

        self.input_detections = self.request.get_param("inputDetections")
        self.input_distances = self.request.get_param("inputDistances")

        self.tracking_history = self.bootstrap.get("tracking_history", {})
        self.pair_durations = self.bootstrap.get("pair_durations", {})
        self.active_groups = self.bootstrap.get("active_groups", {})
        self.frame_count = self.bootstrap.get("frame_count", 0)
        self.last_timestamp_ms = self.bootstrap.get("last_timestamp_ms", 0)

        self.groups_output = []
        self.stats_output = {}

        self.state_manager = StateManager(self.tracking_history, self.pair_durations)
        self.event_accumulator = EventAccumulator()
        self.classifier = GroupClassifier(
            family_age_gap=float(self.family_age_gap),
            couple_mode=self._resolve_option_value(self.couple_mode),
            min_friend_group_size=int(self.min_friend_group_size),
        )

    @staticmethod
    def bootstrap(config: dict) -> dict:
        del config
        return {
            "tracking_history": {},
            "pair_durations": {},
            "active_groups": {},
            "frame_count": 0,
            "last_timestamp_ms": 0,
        }

    @staticmethod
    def _resolve_option_value(option):
        if hasattr(option, "value"):
            return option.value
        if isinstance(option, dict):
            return option.get("value")
        return option

    def _normalize_as_list(self, value):
        if value is None:
            return []
        return value if isinstance(value, list) else [value]

    def _parse_detections(self):
        persons = []
        for det in self._normalize_as_list(self.input_detections):
            d = det if isinstance(det, dict) else det.dict()
            if str(d.get("classLabel", "")).lower() != "person":
                continue
            person_id = d.get("tracking_id", d.get("classId"))
            if person_id is None:
                continue
            persons.append(
                {
                    "id": str(person_id),
                    "bbox": d.get("boundingBox") or {},
                    "timestamp_ms": int(d.get("timestamp_ms", 0)),
                    "demographics": {
                        "age_group": d.get("age_group"),
                        "age_estimate": d.get("age_estimate"),
                        "gender_estimate": d.get("gender_estimate"),
                        "demo_confidence": float(d.get("demo_confidence", 0.0)),
                    },
                    "trajectory": {
                        "velocity_mps": float(d.get("velocity_mps", 0.0)),
                        "heading_deg": float(d.get("heading_deg", 0.0)),
                    },
                    "zone_id": d.get("zone_id"),
                }
            )
        return persons

    def _parse_distances(self):
        dist_map = {}
        for entry in self._normalize_as_list(self.input_distances):
            d = entry if isinstance(entry, dict) else entry.dict()

            id_a = d.get("source_id_a") or d.get("objectIdA")
            id_b = d.get("source_id_b") or d.get("objectIdB")
            distance_cm = d.get("distanceCm", d.get("distance_cm", 0))

            if id_a is None or id_b is None:
                pair_ids = d.get("pair_ids")
                if isinstance(pair_ids, (list, tuple)) and len(pair_ids) == 2:
                    id_a, id_b = pair_ids[0], pair_ids[1]

            if id_a is None or id_b is None:
                continue

            distance_m = float(distance_cm) / 100.0 if float(distance_cm) > 0 else 0.0
            if distance_m <= 0:
                continue
            key = tuple(sorted([str(id_a), str(id_b)]))
            dist_map[key] = distance_m

        return dist_map

    def _cluster_by_distance(self, persons, dist_map):
        if not persons:
            return []

        parent = {p["id"]: p["id"] for p in persons}

        def find(node):
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(a, b):
            root_a, root_b = find(a), find(b)
            if root_a != root_b:
                parent[root_a] = root_b

        ids = [p["id"] for p in persons]
        for idx_a, idx_b in combinations(range(len(ids)), 2):
            key = tuple(sorted([ids[idx_a], ids[idx_b]]))
            if key in dist_map and dist_map[key] <= self.group_distance_m:
                union(ids[idx_a], ids[idx_b])

        clusters = {}
        for person in persons:
            root = find(person["id"])
            clusters.setdefault(root, []).append(person)
        return [members for members in clusters.values() if len(members) >= 2]

    def _score_cohesion(self, clusters, dist_map):
        scored = []
        for members in clusters:
            pair_distances = []
            for idx_a, idx_b in combinations(range(len(members)), 2):
                key = tuple(sorted([members[idx_a]["id"], members[idx_b]["id"]]))
                if key in dist_map:
                    pair_distances.append(dist_map[key])

            if pair_distances:
                avg_distance = sum(pair_distances) / len(pair_distances)
                distance_score = max(0.0, 1.0 - (avg_distance / max(self.group_distance_m, 1e-6)))
            else:
                avg_distance = self.group_distance_m
                distance_score = 0.0

            headings = [float(m["trajectory"].get("heading_deg", 0.0)) for m in members]
            heading_spread = (max(headings) - min(headings)) if headings else 180.0
            heading_spread = min(heading_spread, 360.0 - heading_spread) if heading_spread > 180.0 else heading_spread
            trajectory_score = max(0.0, 1.0 - (heading_spread / 180.0))
            cohesion_score = ((1.0 - self.trajectory_weight) * distance_score) + (self.trajectory_weight * trajectory_score)

            scored.append(
                {
                    "members": members,
                    "cohesion_score": max(0.0, min(1.0, cohesion_score)),
                    "avg_intra_distance_m": avg_distance,
                }
            )
        return scored

    def _filter_by_cohesion_window(self, scored_clusters):
        stable = []
        for scored in scored_clusters:
            members = scored["members"]
            min_duration = float("inf")
            for idx_a, idx_b in combinations(range(len(members)), 2):
                duration = self.state_manager.get_pair_duration(members[idx_a]["id"], members[idx_b]["id"])
                min_duration = min(min_duration, duration)

            if min_duration >= self.cohesion_window_s or scored["cohesion_score"] >= 0.6:
                stable.append(scored)
        return stable

    def _extract_events(self, persons):
        return self.event_accumulator.process(
            tracking_history=self.tracking_history,
            pair_durations=self.pair_durations,
            persons=persons,
            cohesion_window_s=self.cohesion_window_s,
        )

    def _cluster_pair_events(self, members, events):
        relevant = {}
        member_ids = [m["id"] for m in members]
        for idx_a, idx_b in combinations(range(len(member_ids)), 2):
            key = tuple(sorted([member_ids[idx_a], member_ids[idx_b]]))
            if key in events:
                relevant[key] = events[key]
        return relevant

    def _classify_groups(self, stable_clusters, events, current_ts):
        groups = []
        for scored in stable_clusters:
            members = scored["members"]
            member_ids = sorted([m["id"] for m in members])
            pair_events = self._cluster_pair_events(members, events)
            relation_type = self.classifier.classify(
                cluster=members,
                events=pair_events,
                config={
                    "couple_mode": self._resolve_option_value(self.couple_mode),
                    "family_age_gap": self.family_age_gap,
                    "min_friend_group_size": self.min_friend_group_size,
                },
            )

            group_key = "grp_" + uuid.uuid5(uuid.NAMESPACE_DNS, "|".join(member_ids)).hex[:8]
            previous = self.active_groups.get(group_key, {})
            entry_ts = previous.get("entry_timestamp_ms", current_ts)
            confidence = self.classifier.compute_confidence(
                cluster=members,
                pair_events=pair_events,
                cohesion_score=scored["cohesion_score"],
            )
            age_groups = [m["demographics"].get("age_group") for m in members if m["demographics"].get("age_group")]
            zone_candidates = [m.get("zone_id") for m in members if m.get("zone_id")]
            zone_id = zone_candidates[0] if zone_candidates else None

            groups.append(
                {
                    "group_id": group_key,
                    "relation_type": relation_type,
                    "member_ids": member_ids,
                    "demographic_summary": {"age_groups": age_groups, "size": len(member_ids)},
                    "avg_intra_distance_m": round(scored["avg_intra_distance_m"], 4),
                    "cohesion_score": round(scored["cohesion_score"], 4),
                    "confidence": round(confidence, 4),
                    "entry_timestamp_ms": int(entry_ts),
                    "dwell_s": round(max(0.0, (current_ts - entry_ts) / 1000.0), 3),
                    "zone_id": zone_id,
                }
            )

            if relation_type == "family":
                for parent_id, child_id in self.classifier.extract_parent_child_pairs(members):
                    groups.append(
                        {
                            "group_id": "grp_" + uuid.uuid5(uuid.NAMESPACE_DNS, f"{parent_id}:{child_id}").hex[:8],
                            "relation_type": "parent:child",
                            "member_ids": [parent_id, child_id],
                            "demographic_summary": {"age_groups": ["adult", "child"], "size": 2},
                            "avg_intra_distance_m": round(scored["avg_intra_distance_m"], 4),
                            "cohesion_score": round(scored["cohesion_score"], 4),
                            "confidence": round(min(1.0, confidence + 0.05), 4),
                            "entry_timestamp_ms": int(entry_ts),
                            "dwell_s": round(max(0.0, (current_ts - entry_ts) / 1000.0), 3),
                            "zone_id": zone_id,
                        }
                    )

        return groups

    def _identify_solos(self, persons, grouped_ids, unreliable, current_ts):
        solos = []
        unreliable_ids = {p["id"] for p in unreliable}
        for person in persons:
            pid = person["id"]
            if pid in grouped_ids:
                continue
            confidence = 0.5 if pid in unreliable_ids else max(0.3, person["demographics"].get("demo_confidence", 0.0))
            solos.append(
                {
                    "group_id": "grp_" + uuid.uuid5(uuid.NAMESPACE_DNS, f"solo:{pid}").hex[:8],
                    "relation_type": "solo_shopper",
                    "member_ids": [pid],
                    "demographic_summary": {
                        "age_groups": [person["demographics"].get("age_group")] if person["demographics"].get("age_group") else [],
                        "size": 1,
                    },
                    "avg_intra_distance_m": 0.0,
                    "cohesion_score": 0.0,
                    "confidence": round(float(confidence), 4),
                    "entry_timestamp_ms": int(current_ts),
                    "dwell_s": 0.0,
                    "zone_id": person.get("zone_id"),
                }
            )
        return solos

    def _compute_stats(self, groups, persons, process_start_ms, current_ts):
        def count_of(rel):
            return len([g for g in groups if g["relation_type"] == rel])

        return {
            "family_count": count_of("family"),
            "couple_count": count_of("couple"),
            "friend_group_count": count_of("friend_group"),
            "solo_count": count_of("solo_shopper"),
            "adult_group_count": count_of("adult_group"),
            "total_persons": len(persons),
            "window_start_ms": int(min([p.get("timestamp_ms", process_start_ms) for p in persons], default=process_start_ms)),
            "window_end_ms": int(current_ts),
            "meta": {
                "processed_at_ms": int(process_start_ms),
                "detection_count": len(persons),
                "groups_found": len(groups),
            },
        }

    def run(self):
        process_start = int(time.time() * 1000)
        persons = self._parse_detections()
        dist_map = self._parse_distances()
        current_ts = max((p["timestamp_ms"] for p in persons), default=process_start)
        delta_s = (current_ts - self.last_timestamp_ms) / 1000.0 if self.last_timestamp_ms > 0 else 0.0

        self.state_manager.update_tracking_history(persons=persons, timestamp_ms=current_ts)
        self.state_manager.update_pair_durations(
            persons=persons,
            dist_map=dist_map,
            delta_s=delta_s,
            threshold_m=self.group_distance_m,
        )
        self.state_manager.cleanup_stale(current_ts=current_ts, max_age_ms=300000)

        reliable = [p for p in persons if p["demographics"]["demo_confidence"] >= self.confidence_threshold]
        unreliable = [p for p in persons if p["demographics"]["demo_confidence"] < self.confidence_threshold]

        clusters = self._cluster_by_distance(reliable, dist_map)
        scored = self._score_cohesion(clusters, dist_map)
        stable = self._filter_by_cohesion_window(scored)
        events = self._extract_events(persons)
        groups = self._classify_groups(stable, events, current_ts)

        grouped_ids = set()
        for group in groups:
            grouped_ids.update(group["member_ids"])
        groups.extend(self._identify_solos(persons, grouped_ids, unreliable, current_ts))

        enabled = [self._resolve_option_value(item) for item in (self.relation_types_enabled or [])]
        if enabled:
            groups = [g for g in groups if g["relation_type"] in enabled]

        stats = self._compute_stats(groups=groups, persons=persons, process_start_ms=process_start, current_ts=current_ts)

        self.bootstrap["tracking_history"] = self.tracking_history
        self.bootstrap["pair_durations"] = self.pair_durations
        self.bootstrap["active_groups"] = {g["group_id"]: g for g in groups if g["relation_type"] != "solo_shopper"}
        self.bootstrap["frame_count"] = self.frame_count + 1
        self.bootstrap["last_timestamp_ms"] = current_ts

        self.groups_output = groups
        self.stats_output = stats
        return build_response_social_group(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
