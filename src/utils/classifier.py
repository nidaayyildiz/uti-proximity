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
            "n_children":    len(children),
            "n_adults":      len(adults),
            "has_male":      any(g in ("man", "male") for g in genders),
            "has_female":    any(g in ("woman", "female") for g in genders),
        }


    # Formation detection (side_by_side vs other)


    @staticmethod
    def _is_side_by_side(m1, m2):
        cy_diff = abs(m1["cy"] - m2["cy"])
        avg_h   = (m1["h"] + m2["h"]) / 2.0
        return cy_diff < 0.35 * avg_h


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

        #  PARENT-CHILD 
        if ctx.pc_enabled:
            p = self._profile(members, ctx.pc_child_age,
                              ctx.pc_hr_enabled, ctx.pc_hr_value)
            if p["n_children"] >= 1 and p["n_adults"] >= 1 and size <= 3:
                return "parent_child"

        #  COUPLE 
        if ctx.couple_enabled and size == 2:
            m1, m2    = members[0], members[1]
            g1        = (m1.get("gender") or "").lower()
            g2        = (m2.get("gender") or "").lower()
            gender_ok = (
                ctx.couple_gender == "any" or
                (ctx.couple_gender == "required" and
                 {g1, g2} & {"man", "male"} and {g1, g2} & {"woman", "female"})
            )
            form_ok   = (
                ctx.couple_formation == "not_required" or
                self._is_side_by_side(m1, m2)
            )
            pair_key  = tuple(sorted([m1["key"], m2["key"]]))
            dur       = pair_durations.get(pair_key, 0.0)
            if dur >= ctx.couple_min_dur and gender_ok and form_ok:
                return "couple"

        #  FRIEND GROUP 
        if ctx.fg_enabled and size >= ctx.fg_min_size:
            heights   = [m["h"] for m in members]
            min_h     = min(heights) or 1
            height_ok = (max(heights) / min_h) < ctx.fg_height_tol

            child_ok  = True
            if ctx.fg_child_tolerance == "skip":
                child_ok = not any(
                    m.get("age") is not None and int(m["age"]) < ctx.fg_child_age
                    for m in members
                )

            if height_ok and child_ok:
                return "friend_group"

        return None
