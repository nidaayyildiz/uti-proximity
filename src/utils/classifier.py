import math


class Classifier:
    def __init__(self, ctx):
        self.ctx = ctx


    # Child detection


    def _is_child(self, member, child_age, hr_enabled, hr_value):
        age = member.get("age")
        if age is not None:
            return int(age) < child_age
        if hr_enabled:
            return member.get("height_ratio_to_max", 1.0) >= hr_value
        return False

    def _profile(self, members, child_age, hr_enabled, hr_value):
        children = [m for m in members if self._is_child(m, child_age, hr_enabled, hr_value)]
        adults   = [m for m in members if not self._is_child(m, child_age, hr_enabled, hr_value)]
        genders  = {(m.get("gender") or "").lower() for m in adults}
        return {
            "n_children": len(children),
            "n_adults":   len(adults),
            "has_male":   any(g in ("man", "male")     for g in genders),
            "has_female": any(g in ("woman", "female") for g in genders),
        }


    # Formation detection


    @staticmethod
    def _is_side_by_side(m1, m2):
        cy_diff = abs(m1["cy"] - m2["cy"])
        avg_h   = (m1["h"] + m2["h"]) / 2.0
        return cy_diff < 0.35 * avg_h


    # Pose estimation helpers (COCO 17-keypoint)


    @staticmethod
    def _kp_dist(kp1, kp2):
        if not kp1 or not kp2:
            return None
        return math.hypot(kp1["cx"] - kp2["cx"], kp1["cy"] - kp2["cy"])

    def _wrist_proximity(self, m1, m2):
        """Min distance among all 4 wrist combinations, normalised by avg bbox height."""
        kp1  = m1.get("keypoints") or []
        kp2  = m2.get("keypoints") or []
        wa_l = kp1[9]  if len(kp1) > 9  else None
        wa_r = kp1[10] if len(kp1) > 10 else None
        wb_l = kp2[9]  if len(kp2) > 9  else None
        wb_r = kp2[10] if len(kp2) > 10 else None
        valid = [
            d for d in [
                self._kp_dist(wa_l, wb_r),
                self._kp_dist(wa_r, wb_l),
                self._kp_dist(wa_l, wb_l),
                self._kp_dist(wa_r, wb_r),
            ] if d is not None
        ]
        if not valid:
            return None
        avg_h = (m1["h"] + m2["h"]) / 2.0
        return min(valid) / avg_h if avg_h > 0 else None

    def _shoulder_parallelism(self, m1, m2):
        """Angle (degrees) between the two shoulder vectors; small → facing same direction."""
        kp1 = m1.get("keypoints") or []
        kp2 = m2.get("keypoints") or []
        ls1 = kp1[5] if len(kp1) > 5 else None
        rs1 = kp1[6] if len(kp1) > 6 else None
        ls2 = kp2[5] if len(kp2) > 5 else None
        rs2 = kp2[6] if len(kp2) > 6 else None
        if not all([ls1, rs1, ls2, rs2]):
            return None
        v1   = (rs1["cx"] - ls1["cx"], rs1["cy"] - ls1["cy"])
        v2   = (rs2["cx"] - ls2["cx"], rs2["cy"] - ls2["cy"])
        mag1 = math.hypot(*v1)
        mag2 = math.hypot(*v2)
        if mag1 == 0 or mag2 == 0:
            return None
        cos_a = max(-1.0, min(1.0, (v1[0]*v2[0] + v1[1]*v2[1]) / (mag1 * mag2)))
        return math.degrees(math.acos(cos_a))

    def _hip_proximity(self, m1, m2):
        """Distance between hip centres, normalised by avg bbox height."""
        kp1  = m1.get("keypoints") or []
        kp2  = m2.get("keypoints") or []
        lh1  = kp1[11] if len(kp1) > 11 else None
        rh1  = kp1[12] if len(kp1) > 12 else None
        lh2  = kp2[11] if len(kp2) > 11 else None
        rh2  = kp2[12] if len(kp2) > 12 else None
        pts1 = [h for h in [lh1, rh1] if h]
        pts2 = [h for h in [lh2, rh2] if h]
        if not pts1 or not pts2:
            return None
        cx1   = sum(h["cx"] for h in pts1) / len(pts1)
        cy1   = sum(h["cy"] for h in pts1) / len(pts1)
        cx2   = sum(h["cx"] for h in pts2) / len(pts2)
        cy2   = sum(h["cy"] for h in pts2) / len(pts2)
        avg_h = (m1["h"] + m2["h"]) / 2.0
        return math.hypot(cx2 - cx1, cy2 - cy1) / avg_h if avg_h > 0 else None

    def _mutual_gaze(self, m1, m2):
        """True if at least one person's gaze vector points toward the other."""
        def _gazes_toward(kp, target_cx, target_cy):
            nose = kp[0] if len(kp) > 0 else None
            le   = kp[1] if len(kp) > 1 else None
            re   = kp[2] if len(kp) > 2 else None
            if not all([nose, le, re]):
                return False
            eye_cx = (le["cx"] + re["cx"]) / 2
            eye_cy = (le["cy"] + re["cy"]) / 2
            gaze   = (nose["cx"] - eye_cx,      nose["cy"] - eye_cy)
            to_tgt = (target_cx - nose["cx"],    target_cy - nose["cy"])
            mg     = math.hypot(*gaze)
            mt     = math.hypot(*to_tgt)
            if mg == 0 or mt == 0:
                return False
            cos_a = max(-1.0, min(1.0, (gaze[0]*to_tgt[0] + gaze[1]*to_tgt[1]) / (mg * mt)))
            return math.degrees(math.acos(cos_a)) < 45

        kp1 = m1.get("keypoints") or []
        kp2 = m2.get("keypoints") or []
        return _gazes_toward(kp1, m2["cx"], m2["cy"]) or _gazes_toward(kp2, m1["cx"], m1["cy"])

    def _pose_couple_score(self, m1, m2, pair_dur, mixed_gender):
        """
        Weighted score in [0, 1]. Threshold > 0.6 → couple.
        Weights: wrist(0.35) shoulder(0.15) hip(0.15) gaze(0.10) duration(0.15) gender(0.10)
        """
        score = 0.0

        wrist = self._wrist_proximity(m1, m2)
        if wrist is not None and wrist < 0.1:
            score += 0.35

        shoulder = self._shoulder_parallelism(m1, m2)
        if shoulder is not None and shoulder < 30:
            score += 0.15

        hip = self._hip_proximity(m1, m2)
        if hip is not None and hip < 0.3:
            score += 0.15

        if self._mutual_gaze(m1, m2):
            score += 0.10

        if pair_dur >= self.ctx.couple_min_dur:
            score += 0.15

        if mixed_gender:
            score += 0.10

        return score


    # Main classify entry point


    def classify(self, members, pair_durations, formation):
        ctx  = self.ctx
        size = len(members)

        # FAMILY
        if ctx.family_enabled:
            p = self._profile(members, ctx.family_child_age,
                              ctx.family_hr_enabled, ctx.family_hr_value)
            gender_ok = (
                (ctx.family_gender == "both"   and p["has_male"] and p["has_female"]) or
                (ctx.family_gender == "male"   and p["has_male"]) or
                (ctx.family_gender == "female" and p["has_female"])
            )
            if (p["n_children"] >= ctx.family_min_children and
                    p["n_adults"] >= ctx.family_min_adults and gender_ok):
                return "family"

        # PARENT-CHILD
        if ctx.pc_enabled:
            p = self._profile(members, ctx.pc_child_age,
                              ctx.pc_hr_enabled, ctx.pc_hr_value)
            if p["n_children"] >= 1 and p["n_adults"] >= 1 and size <= 3:
                return "parent_child"

        # COUPLE
        if ctx.couple_enabled and size == 2:
            m1, m2       = members[0], members[1]
            g1           = (m1.get("gender") or "").lower()
            g2           = (m2.get("gender") or "").lower()
            mixed_gender = bool(
                {g1, g2} & {"man", "male"} and {g1, g2} & {"woman", "female"}
            )
            gender_ok = (
                ctx.couple_gender == "any" or
                (ctx.couple_gender == "required" and mixed_gender)
            )
            pair_key = tuple(sorted([m1["key"], m2["key"]]))
            dur      = pair_durations.get(pair_key, 0.0)

            if gender_ok:
                if ctx.couple_pose_enabled:
                    if self._pose_couple_score(m1, m2, dur, mixed_gender) > 0.6:
                        return "couple"
                else:
                    form_ok = (
                        ctx.couple_formation == "not_required" or
                        self._is_side_by_side(m1, m2)
                    )
                    if dur >= ctx.couple_min_dur and form_ok:
                        return "couple"

        # FRIEND GROUP
        if ctx.fg_enabled and size >= ctx.fg_min_size:
            heights   = [m["h"] for m in members]
            min_h     = min(heights) or 1
            height_ok = (max(heights) / min_h) < ctx.fg_height_tol

            child_ok = True
            if ctx.fg_child_tolerance == "skip":
                child_ok = not any(
                    m.get("age") is not None and int(m["age"]) < ctx.fg_child_age
                    for m in members
                )

            if height_ok and child_ok:
                return "friend_group"

        return None
