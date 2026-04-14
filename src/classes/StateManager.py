from typing import Dict, List, Tuple, Any


class StateManager:
    def __init__(self, tracking_history: Dict[str, List[dict]], pair_durations: Dict[Tuple[str, str], float]):
        self.tracking_history = tracking_history
        self.pair_durations = pair_durations

    def update_tracking_history(self, persons: List[dict], timestamp_ms: int, max_window_s: int = 300) -> None:
        min_keep_ms = max(0, timestamp_ms - (max_window_s * 1000))
        for person in persons:
            person_id = str(person.get("id"))
            if not person_id:
                continue
            bbox = person.get("bbox") or {}
            left = float(bbox.get("left", 0.0))
            top = float(bbox.get("top", 0.0))
            width = float(bbox.get("width", 0.0))
            height = float(bbox.get("height", 0.0))

            observation = {
                "timestamp_ms": timestamp_ms,
                "zone_id": person.get("zone_id"),
                "x": left + (width / 2.0),
                "y": top + (height / 2.0),
                "heading_deg": float((person.get("trajectory") or {}).get("heading_deg", 0.0)),
                "velocity_mps": float((person.get("trajectory") or {}).get("velocity_mps", 0.0)),
            }
            history = self.tracking_history.setdefault(person_id, [])
            history.append(observation)
            self.tracking_history[person_id] = [h for h in history if h["timestamp_ms"] >= min_keep_ms]

    def update_pair_durations(self, persons: List[dict], dist_map: Dict[Tuple[str, str], float], delta_s: float, threshold_m: float) -> None:
        if delta_s <= 0:
            return

        ids = [str(p.get("id")) for p in persons if p.get("id") is not None]
        current_close_pairs = set()
        for idx_a in range(len(ids)):
            for idx_b in range(idx_a + 1, len(ids)):
                key = tuple(sorted([ids[idx_a], ids[idx_b]]))
                if key in dist_map and dist_map[key] <= threshold_m:
                    current_close_pairs.add(key)
                    self.pair_durations[key] = self.pair_durations.get(key, 0.0) + delta_s

        for key in list(self.pair_durations.keys()):
            if key not in current_close_pairs:
                decayed = self.pair_durations[key] - (delta_s * 2.0)
                if decayed <= 0:
                    del self.pair_durations[key]
                else:
                    self.pair_durations[key] = decayed

    def get_pair_duration(self, id_a: Any, id_b: Any) -> float:
        key = tuple(sorted([str(id_a), str(id_b)]))
        return float(self.pair_durations.get(key, 0.0))

    def cleanup_stale(self, current_ts: int, max_age_ms: int) -> None:
        min_keep_ts = current_ts - max_age_ms
        for person_id in list(self.tracking_history.keys()):
            filtered = [h for h in self.tracking_history[person_id] if h["timestamp_ms"] >= min_keep_ts]
            if filtered:
                self.tracking_history[person_id] = filtered
            else:
                del self.tracking_history[person_id]
