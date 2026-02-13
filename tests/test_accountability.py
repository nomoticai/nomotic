"""Tests for accountability — owner engagement and user activity tracking."""

import time

import pytest

from nomotic.accountability import (
    OwnerActivity,
    OwnerActivityLog,
    UserActivityTracker,
    UserStats,
)


class TestOwnerActivity:
    """OwnerActivity data tests."""

    def test_to_dict(self):
        activity = OwnerActivity(
            timestamp=1000.0,
            owner_id="alice@acme.com",
            activity_type="alert_acknowledged",
            target_agent_id="agent-1",
            detail="Acknowledged drift alert",
        )
        d = activity.to_dict()
        assert d["owner_id"] == "alice@acme.com"
        assert d["activity_type"] == "alert_acknowledged"
        assert d["target_agent_id"] == "agent-1"

    def test_to_dict_minimal(self):
        activity = OwnerActivity(
            timestamp=1000.0,
            owner_id="bob@acme.com",
            activity_type="report_reviewed",
        )
        d = activity.to_dict()
        assert "target_agent_id" not in d
        assert "detail" not in d


class TestOwnerActivityLog:
    """OwnerActivityLog recording and querying tests."""

    def test_record(self):
        log = OwnerActivityLog()
        activity = log.record("alice@acme.com", "alert_acknowledged", target_agent_id="agent-1")
        assert activity.owner_id == "alice@acme.com"
        assert activity.activity_type == "alert_acknowledged"

    def test_get_activities(self):
        log = OwnerActivityLog()
        log.record("alice@acme.com", "alert_acknowledged")
        log.record("alice@acme.com", "config_changed")
        log.record("bob@acme.com", "alert_acknowledged")
        activities = log.get_activities("alice@acme.com")
        assert len(activities) == 2

    def test_get_activities_filter_type(self):
        log = OwnerActivityLog()
        log.record("alice@acme.com", "alert_acknowledged")
        log.record("alice@acme.com", "config_changed")
        activities = log.get_activities("alice@acme.com", activity_type="alert_acknowledged")
        assert len(activities) == 1

    def test_get_activities_filter_since(self):
        log = OwnerActivityLog()
        log.record("alice@acme.com", "alert_acknowledged")
        cutoff = time.time() + 1
        time.sleep(0.01)
        # Future record won't exist - test the filter works
        activities = log.get_activities("alice@acme.com", since=cutoff)
        assert len(activities) == 0

    def test_get_activities_newest_first(self):
        log = OwnerActivityLog()
        log.record("alice@acme.com", "alert_acknowledged", detail="first")
        time.sleep(0.01)
        log.record("alice@acme.com", "config_changed", detail="second")
        activities = log.get_activities("alice@acme.com")
        assert activities[0].detail == "second"

    def test_engagement_score_active(self):
        log = OwnerActivityLog()
        for i in range(15):
            log.record("alice@acme.com", "alert_acknowledged")
        score = log.engagement_score("alice@acme.com")
        assert score["engagement_level"] == "active"
        assert score["total_activities"] == 15
        assert score["activities_in_window"] == 15

    def test_engagement_score_absent(self):
        log = OwnerActivityLog()
        score = log.engagement_score("nobody@acme.com")
        assert score["engagement_level"] == "absent"
        assert score["total_activities"] == 0

    def test_engagement_score_passive(self):
        log = OwnerActivityLog()
        for i in range(3):
            log.record("bob@acme.com", "report_reviewed")
        score = log.engagement_score("bob@acme.com")
        assert score["engagement_level"] == "passive"

    def test_summary(self):
        log = OwnerActivityLog()
        log.record("alice@acme.com", "alert_acknowledged")
        log.record("alice@acme.com", "alert_acknowledged")
        log.record("alice@acme.com", "config_changed")
        summary = log.summary("alice@acme.com")
        assert summary["total_activities"] == 3
        assert summary["by_type"]["alert_acknowledged"] == 2
        assert summary["by_type"]["config_changed"] == 1

    def test_max_records_per_owner(self):
        log = OwnerActivityLog(max_records_per_owner=3)
        for i in range(5):
            log.record("alice@acme.com", "report_reviewed", detail=str(i))
        activities = log.get_activities("alice@acme.com")
        assert len(activities) == 3


class TestUserStats:
    """UserStats data tests."""

    def test_denial_rate_zero_interactions(self):
        stats = UserStats(user_id="user-1")
        assert stats.denial_rate == 0.0

    def test_denial_rate(self):
        stats = UserStats(user_id="user-1", total_interactions=10, deny_count=3)
        assert stats.denial_rate == pytest.approx(0.3)

    def test_to_dict(self):
        stats = UserStats(
            user_id="user-1",
            total_interactions=5,
            allow_count=3,
            deny_count=2,
            agents_interacted={"agent-a", "agent-b"},
        )
        d = stats.to_dict()
        assert d["user_id"] == "user-1"
        assert d["denial_rate"] == pytest.approx(0.4)
        assert sorted(d["agents_interacted"]) == ["agent-a", "agent-b"]


class TestUserActivityTracker:
    """UserActivityTracker tests."""

    def test_record_interaction(self):
        tracker = UserActivityTracker()
        stats = tracker.record_interaction("user-1", "agent-1", "ALLOW", "GOVERNANCE.ALLOW")
        assert stats.total_interactions == 1
        assert stats.allow_count == 1

    def test_record_multiple_interactions(self):
        tracker = UserActivityTracker()
        tracker.record_interaction("user-1", "agent-1", "ALLOW", "GOVERNANCE.ALLOW")
        tracker.record_interaction("user-1", "agent-1", "DENY", "GOVERNANCE.DENY")
        tracker.record_interaction("user-1", "agent-2", "ALLOW", "GOVERNANCE.ALLOW")
        stats = tracker.get_stats("user-1")
        assert stats is not None
        assert stats.total_interactions == 3
        assert stats.allow_count == 2
        assert stats.deny_count == 1
        assert len(stats.agents_interacted) == 2

    def test_get_stats_none(self):
        tracker = UserActivityTracker()
        assert tracker.get_stats("unknown") is None

    def test_classify_normal(self):
        tracker = UserActivityTracker()
        tracker.record_interaction("user-1", "agent-1", "ALLOW", "GOVERNANCE.ALLOW")
        result = tracker.classify_request("user-1", "agent-1", "ALLOW")
        assert result == "normal"

    def test_classify_out_of_scope(self):
        tracker = UserActivityTracker()
        tracker.record_interaction("user-1", "agent-1", "ALLOW", "GOVERNANCE.ALLOW")
        result = tracker.classify_request("user-1", "agent-1", "DENY")
        assert result == "out_of_scope"

    def test_classify_boundary_testing(self):
        tracker = UserActivityTracker()
        for _ in range(5):
            tracker.record_interaction("user-1", "agent-1", "DENY", "GOVERNANCE.DENY")
        result = tracker.classify_request("user-1", "agent-1", "DENY")
        assert result == "boundary_testing"

    def test_classify_suspicious(self):
        tracker = UserActivityTracker()
        for _ in range(6):
            tracker.record_interaction("user-1", "agent-1", "DENY", "GOVERNANCE.DENY")
        for _ in range(4):
            tracker.record_interaction("user-1", "agent-1", "ALLOW", "GOVERNANCE.ALLOW")
        result = tracker.classify_request("user-1", "agent-1", "DENY")
        assert result == "suspicious"

    def test_classify_unknown_user(self):
        tracker = UserActivityTracker()
        result = tracker.classify_request("unknown", "agent-1", "ALLOW")
        assert result == "normal"

    def test_get_flagged_users(self):
        tracker = UserActivityTracker()
        for _ in range(5):
            tracker.record_interaction("bad-user", "agent-1", "DENY", "GOVERNANCE.DENY")
        tracker.record_interaction("good-user", "agent-1", "ALLOW", "GOVERNANCE.ALLOW")
        flagged = tracker.get_flagged_users(threshold=5)
        assert "bad-user" in flagged
        assert "good-user" not in flagged

    def test_max_users_eviction(self):
        tracker = UserActivityTracker(max_users=3)
        # First user — oldest
        tracker.record_interaction("user-0", "agent-1", "ALLOW", "GOVERNANCE.ALLOW")
        time.sleep(0.01)
        tracker.record_interaction("user-1", "agent-1", "ALLOW", "GOVERNANCE.ALLOW")
        time.sleep(0.01)
        tracker.record_interaction("user-2", "agent-1", "ALLOW", "GOVERNANCE.ALLOW")
        time.sleep(0.01)
        # This should evict user-0 (oldest)
        tracker.record_interaction("user-3", "agent-1", "ALLOW", "GOVERNANCE.ALLOW")
        assert tracker.get_stats("user-0") is None
        assert tracker.get_stats("user-3") is not None

    def test_out_of_scope_count(self):
        tracker = UserActivityTracker()
        tracker.record_interaction("user-1", "agent-1", "DENY", "GOVERNANCE.VETO")
        stats = tracker.get_stats("user-1")
        assert stats is not None
        assert stats.out_of_scope_count == 1

    def test_escalate_count(self):
        tracker = UserActivityTracker()
        tracker.record_interaction("user-1", "agent-1", "ESCALATE", "GOVERNANCE.ESCALATE")
        stats = tracker.get_stats("user-1")
        assert stats is not None
        assert stats.escalate_count == 1
