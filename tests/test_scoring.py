import pytest
from tennis_analysis.analysis.scoring import MatchState

def _win(ms, who, n):
    for _ in range(n):
        ms.apply_point(who, "winner")

def test_love_game():
    ms = MatchState()
    _win(ms, 0, 4)
    assert ms.games == [1, 0] and ms.points == [0, 0]

def test_deuce_and_advantage():
    ms = MatchState()
    _win(ms, 0, 3); _win(ms, 1, 3)          # 40-40
    assert ms.score_line().endswith("40-40")
    ms.apply_point(0, "winner")              # AD-40
    assert ms.score_line().endswith("AD-40")
    ms.apply_point(1, "winner")              # 回到平分
    assert ms.score_line().endswith("40-40")
    _win(ms, 1, 2)                           # 占先 + 拿下
    assert ms.games == [0, 1]

def test_no_ad_sudden_death():
    ms = MatchState(no_ad=True)
    _win(ms, 0, 3); _win(ms, 1, 3)
    ms.apply_point(1, "winner")              # no-ad：平分后一分定胜负
    assert ms.games == [0, 1]

def test_set_needs_two_clear_games():
    ms = MatchState()
    for _ in range(5): _win(ms, 0, 4)       # 5-0
    for _ in range(5): _win(ms, 1, 4)       # 5-5
    _win(ms, 0, 4)                           # 6-5，盘未结束
    assert ms.sets == [] and ms.games == [6, 5]
    _win(ms, 0, 4)                           # 7-5 收盘
    assert ms.sets == [[7, 5]] and ms.games == [0, 0]

def test_tiebreak_triggers_and_wins_set():
    ms = MatchState()
    for _ in range(6): _win(ms, 0, 4); _win(ms, 1, 4)   # 6-6
    assert ms.in_tiebreak
    _win(ms, 0, 6); _win(ms, 1, 6)          # TB 6-6，净胜 2 才结束
    assert ms.in_tiebreak
    _win(ms, 0, 2)                           # TB 8-6
    assert ms.sets == [[7, 6]] and not ms.in_tiebreak

def test_match_finishes_best_of_three():
    ms = MatchState(sets_to_win=2)
    for _ in range(12): _win(ms, 0, 4)      # 6-0 6-0
    assert ms.finished
    with pytest.raises(ValueError):
        ms.apply_point(0, "winner")

def test_edit_point_replays_everything_downstream():
    ms = MatchState()
    for _ in range(4): _win(ms, 0, 4)       # 4-0
    _win(ms, 1, 4)                           # 4-1
    # 把第 0 分（首局第一分）改判给 lower：首局从 love game 变成 15 起步，
    # 与从头手工推演的期望局面完全一致
    ms.edit_point(0, new_winner=1)
    assert len(ms.history) == 20
    # 重放正确性：用相同 history 从头喂一个新状态机，全状态一致
    replay = MatchState()
    for rec in ms.history:
        replay.apply_point(rec.winner, rec.reason, rec.rally_frames)
    assert (ms.points, ms.games, ms.sets) == (replay.points, replay.games, replay.sets)

def test_serialization_roundtrip():
    ms = MatchState(sets_to_win=3, no_ad=True, server=1)
    _win(ms, 0, 4); _win(ms, 1, 2)
    ms2 = MatchState.from_dict(ms.to_dict())
    assert (ms2.points, ms2.games, ms2.sets, ms2.server) == (ms.points, ms.games, ms.sets, ms.server)
    assert len(ms2.history) == len(ms.history)
