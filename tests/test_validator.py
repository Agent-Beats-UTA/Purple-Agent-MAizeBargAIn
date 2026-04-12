"""
Unit tests for the M1-M5 validator and heuristic fallback.
Run: pytest tests/ -v
"""

from agent.negotiation import (
    clamp_offer,
    is_offer_feasible,
    my_value_of_their_offer,
    my_value_of_my_offer,
    target_value_for_round,
    validate_action,
    value_of,
)


QUANTITIES = [7, 4, 1]
VALS = [10.0, 20.0, 30.0]  # item 2 is most valued per-unit
BATNA = 50.0


class TestOfferMath:
    def test_value_of(self):
        assert value_of([1, 2, 1], VALS) == 10 + 40 + 30

    def test_my_value_of_their_offer(self):
        # They take [3,2,0] -> I get [4,2,1] -> value = 40+40+30 = 110
        assert my_value_of_their_offer([3, 2, 0], QUANTITIES, VALS) == 110

    def test_clamp_offer(self):
        assert clamp_offer([10, -1, 5], QUANTITIES) == [7, 0, 1]

    def test_is_offer_feasible(self):
        assert is_offer_feasible([7, 4, 1], QUANTITIES)
        assert not is_offer_feasible([8, 0, 0], QUANTITIES)
        assert not is_offer_feasible([-1, 0, 0], QUANTITIES)
        assert not is_offer_feasible([1, 1], QUANTITIES)  # wrong length


class TestValidateCounteroffer:
    def base_kwargs(self, **overrides):
        kw = dict(
            quantities=QUANTITIES,
            my_valuations=VALS,
            batna=BATNA,
            my_previous_offer=None,
            their_last_offer=None,
            is_last_round=False,
        )
        kw.update(overrides)
        return kw

    def test_clean_counteroffer_passes(self):
        # [3,2,0] worth 30+40+0 = 70, above BATNA 50, not extreme
        r = validate_action(
            {"action": "COUNTEROFFER", "offer": [3, 2, 0]},
            **self.base_kwargs(),
        )
        assert r.ok, r.violations

    def test_m1_walks_back_previous_offer(self):
        # Previous kept [4,3,1] worth 40+60+30=130
        # New keeps [3,2,0] worth 70 < 130
        r = validate_action(
            {"action": "COUNTEROFFER", "offer": [3, 2, 0]},
            **self.base_kwargs(my_previous_offer=[4, 3, 1]),
        )
        assert not r.ok
        assert any("M1" in v for v in r.violations)
        # auto-fix should fall back to previous offer
        assert r.fixed_action == {"action": "COUNTEROFFER", "offer": [4, 3, 1]}

    def test_m2_below_batna(self):
        # [0,1,0] worth 20 < BATNA 50
        r = validate_action(
            {"action": "COUNTEROFFER", "offer": [0, 1, 0]},
            **self.base_kwargs(),
        )
        assert not r.ok
        assert any("M2" in v for v in r.violations)
        # auto-fix should bump up to meet BATNA
        assert r.fixed_action is not None
        fixed = r.fixed_action["offer"]
        assert value_of(fixed, VALS) >= BATNA

    def test_m3_takes_everything(self):
        r = validate_action(
            {"action": "COUNTEROFFER", "offer": [7, 4, 1]},
            **self.base_kwargs(),
        )
        assert not r.ok
        assert any("M3" in v for v in r.violations)

    def test_m3_takes_nothing(self):
        r = validate_action(
            {"action": "COUNTEROFFER", "offer": [0, 0, 0]},
            **self.base_kwargs(),
        )
        assert not r.ok
        assert any("M3" in v for v in r.violations)

    def test_clamp_out_of_range(self):
        # Over-quantity values get clamped, no other violations
        r = validate_action(
            {"action": "COUNTEROFFER", "offer": [10, 2, 0]},
            **self.base_kwargs(),
        )
        # Clamped version [7,2,0] is fine, but validator also detects
        # the original was infeasible and returns a fixed_action
        assert r.fixed_action is not None
        assert r.fixed_action["offer"] == [7, 2, 0]


class TestValidateAccept:
    def base(self, **o):
        k = dict(
            quantities=QUANTITIES, my_valuations=VALS, batna=BATNA,
            my_previous_offer=None, their_last_offer=None, is_last_round=False,
        )
        k.update(o)
        return k

    def test_m4_accept_below_batna(self):
        # They take [6,3,1] -> I get [1,1,0] worth 10+20+0=30 < 50
        r = validate_action(
            {"action": "ACCEPT"},
            **self.base(their_last_offer=[6, 3, 1]),
        )
        assert not r.ok
        assert any("M4" in v for v in r.violations)

    def test_accept_above_batna_ok(self):
        # They take [3,2,0] -> I get [4,2,1] worth 110
        r = validate_action(
            {"action": "ACCEPT"},
            **self.base(their_last_offer=[3, 2, 0]),
        )
        assert r.ok

    def test_accept_without_offer_fails(self):
        r = validate_action({"action": "ACCEPT"}, **self.base())
        assert not r.ok
        assert any("PROTOCOL" in v for v in r.violations)


class TestValidateWalk:
    def base(self, **o):
        k = dict(
            quantities=QUANTITIES, my_valuations=VALS, batna=BATNA,
            my_previous_offer=None, their_last_offer=None, is_last_round=False,
        )
        k.update(o)
        return k

    def test_m5_walking_from_good_offer(self):
        # They take [3,2,0] -> I get [4,2,1] worth 110 > BATNA 50
        r = validate_action(
            {"action": "WALK"},
            **self.base(their_last_offer=[3, 2, 0]),
        )
        assert not r.ok
        assert any("M5" in v for v in r.violations)
        # auto-fix: accept instead
        assert r.fixed_action == {"action": "ACCEPT"}

    def test_walk_from_bad_offer_ok(self):
        # They take [7,4,0] -> I get [0,0,1] worth 30 < BATNA 50
        r = validate_action(
            {"action": "WALK"},
            **self.base(their_last_offer=[7, 4, 0]),
        )
        assert r.ok


class TestConcessionSchedule:
    def test_round_zero_is_high(self):
        v = target_value_for_round(0, 5, max_my_value=200.0, batna=50.0)
        assert v > 150

    def test_last_round_near_batna(self):
        v = target_value_for_round(4, 5, max_my_value=200.0, batna=50.0)
        assert 50.0 <= v <= 70.0

    def test_monotonically_decreasing(self):
        vals = [
            target_value_for_round(r, 5, 200.0, 50.0) for r in range(5)
        ]
        assert vals == sorted(vals, reverse=True)

    def test_never_below_batna(self):
        for r in range(10):
            assert target_value_for_round(r, 10, 200.0, 50.0) >= 50.0


class TestParser:
    def test_parse_plain_json(self):
        from core.parser import parse_incoming
        obs = parse_incoming(
            '{"role":"row","round":2,"valuations":[45,72,33],'
            '"batna":85,"quantities":[7,4,1],"last_offer":[3,2,0]}'
        )
        assert obs["role"] == "row"
        assert obs["batna"] == 85

    def test_parse_fenced_json(self):
        from core.parser import parse_incoming
        obs = parse_incoming(
            '```json\n{"role":"col","valuations":[1,2,3],"batna":10}\n```'
        )
        assert obs["role"] == "col"

    def test_parse_alias_keys(self):
        from core.parser import parse_incoming
        obs = parse_incoming('{"my_valuations":[1,2,3],"outside_option":50}')
        assert obs["valuations"] == [1, 2, 3]
        assert obs["batna"] == 50

    def test_parse_llm_action(self):
        from core.parser import parse_llm_action
        a = parse_llm_action('{"action": "COUNTEROFFER", "offer": [3, 2, 0]}')
        assert a == {"action": "COUNTEROFFER", "offer": [3, 2, 0]}

    def test_parse_llm_action_with_prose(self):
        from core.parser import parse_llm_action
        a = parse_llm_action(
            'Thinking about this... {"action":"ACCEPT"} there.'
        )
        assert a == {"action": "ACCEPT"}

    def test_format_action(self):
        from core.parser import format_action
        s = format_action({"action": "COUNTEROFFER", "offer": [1, 2, 0]})
        assert s == '{"action": "COUNTEROFFER", "offer": [1, 2, 0]}'


class TestState:
    def test_update_from_observation(self):
        from agent.state import SessionState
        s = SessionState()
        s.update_from_observation({
            "role": "row", "round": 2,
            "valuations": [10, 20, 30], "batna": 50,
            "quantities": [7, 4, 1], "last_offer": [3, 2, 0],
            "max_rounds": 5,
        })
        assert s.role == "row"
        assert s.current_round == 2
        assert s.my_valuations == [10, 20, 30]
        assert s.batna == 50
        assert s.their_offers == [[3, 2, 0]]

    def test_dedup_their_offer(self):
        from agent.state import SessionState
        s = SessionState()
        s.update_from_observation({"last_offer": [1, 2, 0]})
        s.update_from_observation({"last_offer": [1, 2, 0]})
        assert len(s.their_offers) == 1

    def test_record_my_offer_changes_previous(self):
        from agent.state import SessionState
        s = SessionState()
        assert s.my_previous_offer is None
        s.record_my_offer([3, 2, 0])
        assert s.my_previous_offer == [3, 2, 0]
