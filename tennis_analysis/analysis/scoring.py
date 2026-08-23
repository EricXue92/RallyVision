"""网球计分状态机。核心设计：history 完整保存每一分，edit_point 改任一分后
从头重放全部 history —— 局分/盘分/发球方全部自动重算（SwingVision 不能改历史分，
这是本产品的差异化）。0=upper, 1=lower。"""
from dataclasses import dataclass, field

POINT_LABELS = {0: "0", 1: "15", 2: "30", 3: "40"}

@dataclass
class PointRecord:
    winner: int
    reason: str
    rally_frames: list = field(default_factory=list)

class MatchState:
    def __init__(self, sets_to_win=2, tiebreak_at=6, no_ad=False, server=0):
        self.sets_to_win = sets_to_win
        self.tiebreak_at = tiebreak_at
        self.no_ad = no_ad
        self.initial_server = server
        self.history = []
        self._reset_live()

    def _reset_live(self):
        self.points = [0, 0]
        self.games = [0, 0]
        self.sets = []
        self.server = self.initial_server
        self.in_tiebreak = False
        self.tb_points = [0, 0]
        self.finished = False

    def apply_point(self, winner, reason, rally_frames=None):
        if self.finished:
            raise ValueError("比赛已结束 / match already finished")
        self.history.append(PointRecord(int(winner), reason, list(rally_frames or [])))
        self._score_point(int(winner))

    def edit_point(self, idx, new_winner, reason="manual"):
        old = self.history[idx]                     # 越界让 IndexError 自然抛出
        self.history[idx] = PointRecord(int(new_winner), reason, old.rally_frames)
        replay, self.history = self.history, []
        self._reset_live()
        for rec in replay:
            self.history.append(rec)
            self._score_point(rec.winner)           # 重放绕过 finished 检查：
                                                    # 改分可能让比赛提前结束，其后的分成为无效垃圾分，
                                                    # _score_point 顶部的 finished 短路会静默忽略它们

    def _score_point(self, w):
        if self.finished:
            return
        l = 1 - w
        if self.in_tiebreak:
            self.tb_points[w] += 1
            if self.tb_points[w] >= 7 and self.tb_points[w] - self.tb_points[l] >= 2:
                self.games[w] += 1
                self._close_set()
            elif sum(self.tb_points) % 2 == 1:      # 抢七第 1、3、5…分后换发
                self.server = 1 - self.server
            return
        self.points[w] += 1
        if self.points[w] >= 4 and self.points[w] - self.points[l] >= 2:
            self._win_game(w)
        elif self.no_ad and self.points[w] == 4 and self.points[l] == 3:
            self._win_game(w)                       # no-ad：平分后金球

    def _win_game(self, w):
        self.games[w] += 1
        self.points = [0, 0]
        self.server = 1 - self.server
        l = 1 - w
        if self.games[w] >= 6 and self.games[w] - self.games[l] >= 2:
            self._close_set()
        elif self.games[w] == self.tiebreak_at and self.games[l] == self.tiebreak_at:
            self.in_tiebreak = True
            self.tb_points = [0, 0]

    def _close_set(self):
        self.sets.append(list(self.games))
        self.games = [0, 0]
        self.in_tiebreak = False
        self.tb_points = [0, 0]
        won = [sum(1 for s in self.sets if s[i] > s[1 - i]) for i in (0, 1)]
        if max(won) >= self.sets_to_win:
            self.finished = True

    def score_line(self):
        parts = [" ".join("%d-%d" % (a, b) for a, b in self.sets)]
        parts.append("%d-%d" % (self.games[0], self.games[1]))
        if self.in_tiebreak:
            parts.append("TB %d-%d" % (self.tb_points[0], self.tb_points[1]))
        else:
            p0, p1 = self.points
            if p0 >= 3 and p1 >= 3:
                parts.append("40-40" if p0 == p1 else ("AD-40" if p0 > p1 else "40-AD"))
            else:
                parts.append("%s-%s" % (POINT_LABELS[p0], POINT_LABELS[p1]))
        return " | ".join(p for p in parts if p)

    def to_dict(self):
        return {
            "config": {"sets_to_win": self.sets_to_win, "tiebreak_at": self.tiebreak_at,
                       "no_ad": self.no_ad, "server": self.initial_server},
            "history": [{"winner": r.winner, "reason": r.reason, "rally_frames": r.rally_frames}
                        for r in self.history],
        }

    @classmethod
    def from_dict(cls, data):
        ms = cls(**data["config"])
        for r in data["history"]:
            ms.history.append(PointRecord(r["winner"], r["reason"], r.get("rally_frames", [])))
            ms._score_point(r["winner"])
        return ms
