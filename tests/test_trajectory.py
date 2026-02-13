"""Tests for the trust trajectory system."""

import threading
import time

from nomotic.trajectory import (
    SOURCE_COMPLETION_INTERRUPT,
    SOURCE_COMPLETION_SUCCESS,
    SOURCE_DRIFT_ADJUSTMENT,
    SOURCE_DRIFT_RECOVERY,
    SOURCE_TIME_DECAY,
    SOURCE_VERDICT_ALLOW,
    SOURCE_VERDICT_DENY,
    TrustEvent,
    TrustTrajectory,
)


class TestTrustEvent:
    def test_direction_up(self):
        e = TrustEvent(
            timestamp=1.0, trust_before=0.5, trust_after=0.6,
            delta=0.1, source="test", reason="test",
        )
        assert e.direction == "up"

    def test_direction_down(self):
        e = TrustEvent(
            timestamp=1.0, trust_before=0.6, trust_after=0.5,
            delta=-0.1, source="test", reason="test",
        )
        assert e.direction == "down"

    def test_direction_unchanged(self):
        e = TrustEvent(
            timestamp=1.0, trust_before=0.5, trust_after=0.5001,
            delta=0.0001, source="test", reason="test",
        )
        assert e.direction == "unchanged"

    def test_to_dict(self):
        e = TrustEvent(
            timestamp=1000.0, trust_before=0.5, trust_after=0.6,
            delta=0.1, source="verdict:allow", reason="Action allowed",
            metadata={"key": "val"},
        )
        d = e.to_dict()
        assert d["timestamp"] == 1000.0
        assert d["source"] == "verdict:allow"
        assert d["reason"] == "Action allowed"
        assert d["metadata"] == {"key": "val"}

    def test_to_dict_no_metadata(self):
        e = TrustEvent(
            timestamp=1.0, trust_before=0.5, trust_after=0.6,
            delta=0.1, source="test", reason="test",
        )
        d = e.to_dict()
        assert "metadata" not in d

    def test_frozen(self):
        e = TrustEvent(
            timestamp=1.0, trust_before=0.5, trust_after=0.6,
            delta=0.1, source="test", reason="test",
        )
        try:
            e.delta = 0.2  # type: ignore[misc]
            assert False, "Should be frozen"
        except AttributeError:
            pass


class TestTrustTrajectory:
    def test_record_event(self):
        t = TrustTrajectory("agent-1")
        event = t.record(
            trust_before=0.5, trust_after=0.6,
            source=SOURCE_VERDICT_ALLOW, reason="Allowed",
        )
        assert event is not None
        assert abs(event.delta - 0.1) < 1e-10
        assert event.source == "verdict:allow"
        assert len(t) == 1

    def test_record_skips_insignificant_delta(self):
        t = TrustTrajectory("agent-1")
        event = t.record(
            trust_before=0.5, trust_after=0.5005,
            source="test", reason="tiny change",
        )
        assert event is None
        assert len(t) == 0

    def test_record_at_threshold(self):
        """Delta at or below 0.001 should NOT be recorded."""
        t = TrustTrajectory("agent-1")
        event = t.record(
            trust_before=0.5, trust_after=0.5009,
            source="test", reason="at threshold",
        )
        assert event is None
        assert len(t) == 0

    def test_record_just_above_threshold(self):
        """Delta just above 0.001 should be recorded."""
        t = TrustTrajectory("agent-1")
        event = t.record(
            trust_before=0.5, trust_after=0.502,
            source="test", reason="above threshold",
        )
        assert event is not None
        assert len(t) == 1

    def test_events_returns_copy(self):
        t = TrustTrajectory("agent-1")
        t.record(trust_before=0.5, trust_after=0.6, source="a", reason="r")
        events1 = t.events
        events2 = t.events
        assert events1 == events2
        assert events1 is not events2

    def test_events_chronological_order(self):
        t = TrustTrajectory("agent-1")
        t.record(trust_before=0.5, trust_after=0.6, source="a", reason="first")
        t.record(trust_before=0.6, trust_after=0.7, source="b", reason="second")
        events = t.events
        assert events[0].reason == "first"
        assert events[1].reason == "second"
        assert events[0].timestamp <= events[1].timestamp

    def test_latest_returns_most_recent(self):
        t = TrustTrajectory("agent-1")
        assert t.latest is None
        t.record(trust_before=0.5, trust_after=0.6, source="a", reason="first")
        t.record(trust_before=0.6, trust_after=0.7, source="b", reason="second")
        assert t.latest is not None
        assert t.latest.reason == "second"

    def test_events_since(self):
        t = TrustTrajectory("agent-1")
        t.record(trust_before=0.5, trust_after=0.6, source="a", reason="first")
        cutoff = time.time()
        time.sleep(0.01)
        t.record(trust_before=0.6, trust_after=0.7, source="b", reason="second")
        after = t.events_since(cutoff)
        assert len(after) == 1
        assert after[0].reason == "second"

    def test_events_by_source_exact(self):
        t = TrustTrajectory("agent-1")
        t.record(trust_before=0.5, trust_after=0.6, source=SOURCE_VERDICT_ALLOW, reason="a")
        t.record(trust_before=0.6, trust_after=0.55, source=SOURCE_VERDICT_DENY, reason="b")
        t.record(trust_before=0.55, trust_after=0.65, source=SOURCE_VERDICT_ALLOW, reason="c")
        allows = t.events_by_source(SOURCE_VERDICT_ALLOW)
        assert len(allows) == 2
        denies = t.events_by_source(SOURCE_VERDICT_DENY)
        assert len(denies) == 1

    def test_events_by_source_prefix(self):
        t = TrustTrajectory("agent-1")
        t.record(trust_before=0.5, trust_after=0.6, source=SOURCE_VERDICT_ALLOW, reason="a")
        t.record(trust_before=0.6, trust_after=0.55, source=SOURCE_VERDICT_DENY, reason="b")
        t.record(trust_before=0.55, trust_after=0.53, source=SOURCE_DRIFT_ADJUSTMENT, reason="c")
        verdict_events = t.events_by_source("verdict")
        assert len(verdict_events) == 2
        drift_events = t.events_by_source("drift")
        assert len(drift_events) == 1

    def test_net_delta(self):
        t = TrustTrajectory("agent-1")
        t.record(trust_before=0.5, trust_after=0.6, source="a", reason="up")
        t.record(trust_before=0.6, trust_after=0.55, source="b", reason="down")
        assert abs(t.net_delta - 0.05) < 1e-10

    def test_trend_rising(self):
        t = TrustTrajectory("agent-1")
        for i in range(20):
            base = 0.5 + i * 0.01
            t.record(trust_before=base, trust_after=base + 0.01, source="a", reason="up")
        assert t.trend == "rising"

    def test_trend_falling(self):
        t = TrustTrajectory("agent-1")
        for i in range(20):
            base = 0.9 - i * 0.01
            t.record(trust_before=base, trust_after=base - 0.01, source="a", reason="down")
        assert t.trend == "falling"

    def test_trend_stable_empty(self):
        t = TrustTrajectory("agent-1")
        assert t.trend == "stable"

    def test_trend_volatile(self):
        t = TrustTrajectory("agent-1")
        for i in range(20):
            if i % 2 == 0:
                t.record(trust_before=0.5, trust_after=0.55, source="a", reason="up")
            else:
                t.record(trust_before=0.55, trust_after=0.5, source="b", reason="down")
        assert t.trend == "volatile"

    def test_summary_structure(self):
        t = TrustTrajectory("agent-1")
        t.record(trust_before=0.5, trust_after=0.6, source=SOURCE_VERDICT_ALLOW, reason="a")
        t.record(trust_before=0.6, trust_after=0.55, source=SOURCE_VERDICT_DENY, reason="b")

        s = t.summary()
        assert s["agent_id"] == "agent-1"
        assert s["total_events"] == 2
        assert "net_delta" in s
        assert "trend" in s
        assert "current_trust" in s
        assert "sources" in s
        assert "recent_events" in s

    def test_summary_groups_by_source(self):
        t = TrustTrajectory("agent-1")
        t.record(trust_before=0.5, trust_after=0.51, source=SOURCE_VERDICT_ALLOW, reason="a")
        t.record(trust_before=0.51, trust_after=0.52, source=SOURCE_VERDICT_ALLOW, reason="b")
        t.record(trust_before=0.52, trust_after=0.47, source=SOURCE_VERDICT_DENY, reason="c")

        s = t.summary()
        assert SOURCE_VERDICT_ALLOW in s["sources"]
        assert s["sources"][SOURCE_VERDICT_ALLOW]["count"] == 2
        assert SOURCE_VERDICT_DENY in s["sources"]
        assert s["sources"][SOURCE_VERDICT_DENY]["count"] == 1

    def test_summary_current_trust(self):
        t = TrustTrajectory("agent-1")
        t.record(trust_before=0.5, trust_after=0.6, source="a", reason="r")
        s = t.summary()
        assert s["current_trust"] == 0.6

    def test_summary_empty(self):
        t = TrustTrajectory("agent-1")
        s = t.summary()
        assert s["total_events"] == 0
        assert s["current_trust"] is None

    def test_caps_at_max_events(self):
        t = TrustTrajectory("agent-1", max_events=10)
        for i in range(20):
            t.record(
                trust_before=0.5, trust_after=0.52,
                source="a", reason=f"event-{i}",
            )
        assert len(t) == 10
        # Should keep the latest
        events = t.events
        assert events[-1].reason == "event-19"
        assert events[0].reason == "event-10"

    def test_to_list_serialization(self):
        t = TrustTrajectory("agent-1")
        t.record(trust_before=0.5, trust_after=0.6, source="a", reason="up")
        t.record(trust_before=0.6, trust_after=0.55, source="b", reason="down")
        lst = t.to_list()
        assert len(lst) == 2
        assert isinstance(lst[0], dict)
        assert lst[0]["source"] == "a"
        assert lst[1]["source"] == "b"

    def test_clear(self):
        t = TrustTrajectory("agent-1")
        t.record(trust_before=0.5, trust_after=0.6, source="a", reason="r")
        assert len(t) == 1
        t.clear()
        assert len(t) == 0
        assert t.events == []
        assert t.latest is None

    def test_thread_safety(self):
        t = TrustTrajectory("agent-1", max_events=1000)
        errors: list[Exception] = []

        def writer(offset: int):
            try:
                for i in range(100):
                    t.record(
                        trust_before=0.5, trust_after=0.52,
                        source=f"thread-{offset}", reason=f"event-{i}",
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert len(errors) == 0
        assert len(t) == 500  # 5 threads * 100 events

    def test_agent_id_property(self):
        t = TrustTrajectory("agent-42")
        assert t.agent_id == "agent-42"

    def test_record_with_metadata(self):
        t = TrustTrajectory("agent-1")
        event = t.record(
            trust_before=0.5, trust_after=0.6,
            source="test", reason="with meta",
            metadata={"drift_overall": 0.3},
        )
        assert event is not None
        assert event.metadata == {"drift_overall": 0.3}


class TestSourceConstants:
    def test_source_constants_defined(self):
        assert SOURCE_VERDICT_ALLOW == "verdict:allow"
        assert SOURCE_VERDICT_DENY == "verdict:deny"
        assert SOURCE_COMPLETION_SUCCESS == "completion:success"
        assert SOURCE_COMPLETION_INTERRUPT == "completion:interrupt"
        assert SOURCE_TIME_DECAY == "time_decay"
        assert SOURCE_DRIFT_ADJUSTMENT == "drift:adjustment"
        assert SOURCE_DRIFT_RECOVERY == "drift:recovery"
