from typing import Dict, List, Tuple, Any


class GroupClassifier:
    def __init__(self, family_age_gap: float, couple_mode: str, min_friend_group_size: int):
        self.family_age_gap = family_age_gap
        self.couple_mode = couple_mode
        self.min_friend_group_size = min_friend_group_size

    def classify(self, cluster: List[dict], events: Dict[Tuple[str, str], List[str]], config: Dict[str, Any]) -> str:
        del events
        if config.get("couple_mode"):
            self.couple_mode = config["couple_mode"]
        if config.get("family_age_gap") is not None:
            self.family_age_gap = config["family_age_gap"]
        if config.get("min_friend_group_size") is not None:
            self.min_friend_group_size = config["min_friend_group_size"]
        return self._classify_group_type(cluster)

    def _classify_group_type(self, members: List[dict]) -> str:
        age_groups = [m["demographics"].get("age_group") for m in members if m.get("demographics", {}).get("age_group")]
        age_estimates = [m["demographics"].get("age_estimate") for m in members if m.get("demographics", {}).get("age_estimate") is not None]
        genders = [m["demographics"].get("gender_estimate") for m in members if m.get("demographics", {}).get("gender_estimate")]
        size = len(members)

        has_child = any(ag in ("child", "teen") for ag in age_groups)
        has_adult = any(ag in ("adult", "senior", "young_adult") for ag in age_groups)

        if has_child and has_adult:
            if len(age_estimates) >= 2:
                if max(age_estimates) - min(age_estimates) >= self.family_age_gap:
                    return "family"
            else:
                return "family"

        if size == 2 and has_adult and not has_child:
            if self.couple_mode == "mixed_gender_only":
                if "male" in genders and "female" in genders:
                    return "couple"
            else:
                return "couple"

        if size >= self.min_friend_group_size and age_groups:
            most_common = max(set(age_groups), key=age_groups.count)
            if age_groups.count(most_common) / len(age_groups) >= 0.6:
                return "friend_group"

        if size >= 2 and has_adult:
            return "adult_group"
        return "adult_group"

    def extract_parent_child_pairs(self, family_cluster: List[dict]) -> List[Tuple[str, str]]:
        adults = [p for p in family_cluster if p.get("demographics", {}).get("age_group") in ("adult", "young_adult", "senior")]
        children = [p for p in family_cluster if p.get("demographics", {}).get("age_group") in ("child", "teen")]
        pairs = []
        for adult in adults:
            for child in children:
                pairs.append((str(adult["id"]), str(child["id"])))
        return pairs

    def compute_confidence(self, cluster: List[dict], pair_events: Dict[Tuple[str, str], List[str]], cohesion_score: float) -> float:
        demo_scores = [float(p.get("demographics", {}).get("demo_confidence", 0.0)) for p in cluster]
        demo_conf = sum(demo_scores) / len(demo_scores) if demo_scores else 0.0
        event_count = sum(len(v) for v in pair_events.values())
        event_boost = min(0.2, event_count * 0.02)
        score = (0.6 * cohesion_score) + (0.4 * demo_conf) + event_boost
        return max(0.0, min(1.0, score))
