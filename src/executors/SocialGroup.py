import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.Proximity.src.utils.response import build_response
from components.Proximity.src.utils.classifier import Classifier
from components.Proximity.src.models.PackageModel import PackageModel, Detection, BoundingBox


class SocialGroup(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))

        self.distances = self.request.get_param("inputDistances") or []
        self.persons   = self.request.get_param("inputPersons")   or []
        try:
            self.facial = self.request.get_param("inputFacialAnalysis") or []
        except Exception:
            self.facial = []

        # General
        self.prox_cm  = self.request.get_param("configProximityThresholdCm") or 150.0
        self.prox_dur = self.request.get_param("configProximityDurationSec") or 3.0

        # Classification flags
        self.family_enabled = self.request.get_param("configFamilyDetection")       == "enabled"
        self.pc_enabled     = self.request.get_param("configParentChildDetection")   == "enabled"
        self.couple_enabled = self.request.get_param("configCoupleDetection")        == "enabled"
        self.fg_enabled     = self.request.get_param("configFriendGroupDetection")   == "enabled"

        # Family
        self.family_min_children = self.request.get_param("configFamilyMinChildren") or 1
        self.family_min_adults   = self.request.get_param("configFamilyMinAdults")   or 2
        self.family_gender       = self.request.get_param("configFamilyAdultGender") or "both"
        self.family_child_age    = self.request.get_param("configFamilyChildAge")    or 18
        self.family_hr_enabled   = self.request.get_param("configFamilyHeightRatio") == "enabled"
        self.family_hr_value     = self.request.get_param("configFamilyHRValue")     or 1.4

        # Parent-Child
        self.pc_child_age  = self.request.get_param("configPCChildAge")   or 18
        self.pc_hr_enabled = self.request.get_param("configPCHeightRatio") == "enabled"
        self.pc_hr_value   = self.request.get_param("configPCHRValue")    or 1.4

        # Couple
        self.couple_min_dur      = self.request.get_param("configCoupleMinDuration")    or 10.0
        self.couple_formation    = self.request.get_param("configCoupleFormation")      or "not_required"
        self.couple_gender       = self.request.get_param("configCoupleGender")         or "any"
        self.couple_pose_enabled = self.request.get_param("configCouplePoseEstimation") == "enabled"

        # Friend Group
        self.fg_min_size       = self.request.get_param("configFGMinSize")          or 2
        self.fg_height_tol     = self.request.get_param("configFGHeightTolerance")  or 1.3
        self.fg_child_tolerance = self.request.get_param("configFGChildTolerance") or "allow"
        self.fg_child_age      = self.request.get_param("configFGChildAge")         or 18

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {
            "pair_durations": {},  # {(key1, key2): float seconds}
            "last_time": None,
        }



    def _person_key(self, det):
        tid = det.get("trackerID")
        if tid is not None:
            return f"tid_{int(tid)}"
        bb = det.get("boundingBox") or {}
        cx = int(bb.get("left", 0) + bb.get("width",  0) / 2)
        cy = int(bb.get("top",  0) + bb.get("height", 0) / 2)
        return f"cx_{cx}_{cy}"

    def _build_persons_dict(self):
        """Deduplicate outputObjects by person key → profile dict."""
        persons = {}
        for det in self.persons:
            bb = det.get("boundingBox") or {}
            if not bb:
                continue
            cx  = round(bb.get("left", 0) + bb.get("width",  0) / 2)
            cy  = round(bb.get("top",  0) + bb.get("height", 0) / 2)
            key = self._person_key(det)
            if key not in persons:
                persons[key] = {
                    "key":        key,
                    "bbox":       bb,
                    "h":          bb.get("height", 1),
                    "cx":         cx,
                    "cy":         cy,
                    "trackerID":  det.get("trackerID"),
                    "confidence": det.get("confidence", 0.0),
                    "age":        None,
                    "gender":     None,
                    "height_ratio_to_max": 1.0,
                    "keypoints":  det.get("keyPoints") or [],
                }
        return persons

    def _enrich_with_facial(self, persons):
        """Overlay age/gender from FacialAnalysis; compute height_ratio_to_max."""
        facial_lookup = {}
        for det in self.facial:
            bb = det.get("boundingBox") or {}
            if not bb:
                continue
            cx = round(bb.get("left", 0) + bb.get("width",  0) / 2)
            cy = round(bb.get("top",  0) + bb.get("height", 0) / 2)
            facial_lookup[(cx, cy)] = det

        heights = [p["h"] for p in persons.values() if p["h"] > 0]
        max_h   = max(heights) if heights else 1

        for p in persons.values():
            face = facial_lookup.get((p["cx"], p["cy"]))
            if face:
                p["age"]    = face.get("age")
                p["gender"] = face.get("gender")
            p["height_ratio_to_max"] = max_h / p["h"] if p["h"] > 0 else 1.0



    def _find_person_key(self, persons, cx, cy, tol=5):
        """Match a keyPoint coordinate to a person key within tolerance."""
        for key, p in persons.items():
            if abs(p["cx"] - cx) <= tol and abs(p["cy"] - cy) <= tol:
                return key
        return None

    def _find_proximate_pairs(self, persons):
        """Return set of sorted (key1, key2) pairs where distanceCm < threshold."""
        pairs = set()
        for det in self.distances:
            kps     = det.get("keyPoints") or []
            dist_cm = det.get("distanceCm")
            if len(kps) < 2 or dist_cm is None or dist_cm > self.prox_cm:
                continue
            k1 = self._find_person_key(persons, round(kps[0]["cx"]), round(kps[0]["cy"]))
            k2 = self._find_person_key(persons, round(kps[1]["cx"]), round(kps[1]["cy"]))
            if k1 and k2 and k1 != k2:
                pairs.add(tuple(sorted([k1, k2])))
        return pairs

    def _update_durations(self, current_pairs, dt):
        """Increment duration for active pairs; remove pairs no longer seen."""
        durs = self.bootstrap["pair_durations"]
        for pair in current_pairs:
            durs[pair] = durs.get(pair, 0.0) + dt
        for pair in list(durs):
            if pair not in current_pairs:
                del durs[pair]
        self.bootstrap["pair_durations"] = durs



    def _build_groups(self, active_pairs, persons):
        """Connected components from active proximity pairs."""
        parent = {k: k for k in persons}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for k1, k2 in active_pairs:
            if k1 in parent and k2 in parent:
                union(k1, k2)

        groups = {}
        for k in persons:
            root = find(k)
            groups.setdefault(root, []).append(k)

        return list(groups.values())



    def _pair_durations_for_group(self, member_keys):
        durs = self.bootstrap["pair_durations"]
        result = {}
        keys = list(member_keys)
        for i, k1 in enumerate(keys):
            for k2 in keys[i + 1:]:
                pair = tuple(sorted([k1, k2]))
                result[pair] = durs.get(pair, 0.0)
        return result

    def _make_group_detection(self, members, relation_type):
        lefts   = [p["bbox"].get("left",   0) for p in members]
        tops    = [p["bbox"].get("top",    0) for p in members]
        rights  = [p["bbox"].get("left",   0) + p["bbox"].get("width",  0) for p in members]
        bottoms = [p["bbox"].get("top",    0) + p["bbox"].get("height", 0) for p in members]

        left   = min(lefts)
        top    = min(tops)
        width  = max(rights)  - left
        height = max(bottoms) - top

        det = Detection(
            boundingBox=BoundingBox(left=left, top=top, width=width, height=height),
            confidence=1.0,
            classLabel=relation_type,
            classId=0,
        )
        det.memberCount = len(members)
        return det



    def run(self):
        now  = time.time()
        last = self.bootstrap.get("last_time") or now
        dt   = max(0.0, now - last)
        self.bootstrap["last_time"] = now

        persons = self._build_persons_dict()
        if not persons:
            self.groups = []
            return build_response(context=self)

        self._enrich_with_facial(persons)

        current_pairs = self._find_proximate_pairs(persons)
        self._update_durations(current_pairs, dt)

        active_pairs = {p for p, d in self.bootstrap["pair_durations"].items()
                        if d >= self.prox_dur}

        groups     = self._build_groups(active_pairs, persons)
        classifier = Classifier(self)
        results    = []

        for member_keys in groups:
            if len(member_keys) < 2:
                continue
            members    = [persons[k] for k in member_keys]
            pair_durs  = self._pair_durations_for_group(member_keys)
            relation   = classifier.classify(members, pair_durs, formation=None)
            if relation:
                results.append(self._make_group_detection(members, relation))

        self.groups = results
        return build_response(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
