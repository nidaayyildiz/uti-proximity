from typing import Dict, List, Tuple


class EventAccumulator:
    def process(self, tracking_history: Dict[str, List[dict]], pair_durations: Dict[Tuple[str, str], float], persons: List[dict], cohesion_window_s: int) -> Dict[Tuple[str, str], List[str]]:
        events: Dict[Tuple[str, str], List[str]] = {}
        person_by_id = {str(p["id"]): p for p in persons if p.get("id") is not None}
        ids = list(person_by_id.keys())

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                key = tuple(sorted([ids[i], ids[j]]))
                pair_events: List[str] = []

                history_a = tracking_history.get(ids[i], [])
                history_b = tracking_history.get(ids[j], [])
                pa = person_by_id[ids[i]]
                pb = person_by_id[ids[j]]

                if history_a and history_b:
                    enter_delta = abs(history_a[0]["timestamp_ms"] - history_b[0]["timestamp_ms"])
                    if enter_delta <= 5000:
                        pair_events.append("co_entry")

                    zone_a = [h.get("zone_id") for h in history_a if h.get("zone_id")]
                    zone_b = [h.get("zone_id") for h in history_b if h.get("zone_id")]
                    if zone_a and zone_b and zone_a[-1] == zone_b[-1] and pair_durations.get(key, 0.0) >= cohesion_window_s:
                        pair_events.append("zone_co_presence")

                    heading_diff = abs(float((pa.get("trajectory") or {}).get("heading_deg", 0.0)) - float((pb.get("trajectory") or {}).get("heading_deg", 0.0)))
                    heading_diff = min(heading_diff, 360.0 - heading_diff)
                    if heading_diff < 30.0 and pair_durations.get(key, 0.0) > 10.0:
                        pair_events.append("shared_trajectory")

                    va = float((pa.get("trajectory") or {}).get("velocity_mps", 0.0))
                    vb = float((pb.get("trajectory") or {}).get("velocity_mps", 0.0))
                    if va < 0.2 and vb < 0.2:
                        pair_events.append("synchronized_stop")

                if pair_events:
                    events[key] = pair_events

        return events

    def get_events_for_pair(self, events: Dict[Tuple[str, str], List[str]], id_a: str, id_b: str) -> List[str]:
        key = tuple(sorted([str(id_a), str(id_b)]))
        return events.get(key, [])
