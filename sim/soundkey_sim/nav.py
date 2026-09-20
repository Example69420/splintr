"""Accessible navigation model + simplified input handler.

The navigation model never requires the screen: every move announces the newly
focused item, every selection is confirmed by a cue, and the edges of a list
are signalled so the user is never left guessing. The input handler maps the
ten fixed gestures (from ``cues/gesture_map.json``) onto navigation actions,
identically on every screen.

Author: Krishita Sanjay Choksi
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .sonifier import Sonifier, SoundEvent


@dataclass
class MenuNode:
    label: str
    children: List["MenuNode"] = field(default_factory=list)
    # For leaf actions: a callable(session) -> List[SoundEvent].
    action: Optional[Callable[["NavSession"], List[SoundEvent]]] = None

    @property
    def is_menu(self) -> bool:
        return bool(self.children)


class NavSession:
    """Drives navigation over a menu tree in response to gesture *actions*.

    Actions are the semantic verbs produced by the input handler:
    prev_item, next_item, select, back, go_home, repeat, help, primary_run,
    increase, decrease.
    """

    def __init__(self, root: MenuNode, sonifier: Optional[Sonifier] = None):
        self.root = root
        self.son = sonifier or Sonifier()
        self.stack: List[MenuNode] = [root]
        self.index: List[int] = [0]
        self._last_announcement: List[SoundEvent] = []

    # --- helpers ---------------------------------------------------------

    @property
    def current_menu(self) -> MenuNode:
        return self.stack[-1]

    @property
    def current_item(self) -> MenuNode:
        return self.current_menu.children[self.index[-1]]

    def _announce_current(self) -> List[SoundEvent]:
        item = self.current_item
        ev = [SoundEvent("nav_move", {}, item.label)]
        self._last_announcement = ev
        return ev

    # --- gesture actions -------------------------------------------------

    def do(self, action: str) -> List[SoundEvent]:
        handler = getattr(self, f"_act_{action}", None)
        if handler is None:
            raise ValueError(f"unknown action: {action}")
        return handler()

    def _act_next_item(self) -> List[SoundEvent]:
        if self.index[-1] < len(self.current_menu.children) - 1:
            self.index[-1] += 1
            return self._announce_current()
        return [SoundEvent("nav_boundary", {}, "end of list")]

    def _act_prev_item(self) -> List[SoundEvent]:
        if self.index[-1] > 0:
            self.index[-1] -= 1
            return self._announce_current()
        return [SoundEvent("nav_boundary", {}, "top of list")]

    def _act_select(self) -> List[SoundEvent]:
        item = self.current_item
        if item.is_menu:
            self.stack.append(item)
            self.index.append(0)
            ev = [SoundEvent("select", {}, item.label)]
            ev += self._announce_current()
            self._last_announcement = ev
            return ev
        if item.action is not None:
            ev = [SoundEvent("select", {}, item.label)]
            ev += item.action(self)
            self._last_announcement = ev
            return ev
        return [SoundEvent("error", {}, "nothing to do")]

    def _act_back(self) -> List[SoundEvent]:
        if len(self.stack) > 1:
            self.stack.pop()
            self.index.pop()
            ev = [SoundEvent("back", {}, "back")]
            ev += self._announce_current()
            self._last_announcement = ev
            return ev
        return [SoundEvent("nav_boundary", {}, "already at home")]

    def _act_go_home(self) -> List[SoundEvent]:
        self.stack = [self.root]
        self.index = [0]
        ev = [SoundEvent("home", {}, "home")]
        ev += self._announce_current()
        self._last_announcement = ev
        return ev

    def _act_repeat(self) -> List[SoundEvent]:
        if self._last_announcement:
            return [SoundEvent("repeat", {}, self._last_announcement[-1].caption)]
        return self._announce_current()

    def _act_help(self) -> List[SoundEvent]:
        loc = " / ".join(n.label for n in self.stack)
        return [SoundEvent("repeat", {}, f"you are in {loc}; item {self.index[-1]+1} of "
                                          f"{len(self.current_menu.children)}: {self.current_item.label}")]

    def _act_primary_run(self) -> List[SoundEvent]:
        # By default the primary action is 'select the focused item'.
        return self._act_select()

    def _act_increase(self) -> List[SoundEvent]:
        return self._act_next_item()

    def _act_decrease(self) -> List[SoundEvent]:
        return self._act_prev_item()


class InputHandler:
    """Maps (button, press) -> semantic action using the gesture map."""

    def __init__(self, gesture_map: dict):
        self.table: Dict[tuple, str] = {}
        for g in gesture_map["gestures"]:
            self.table[(g["button"], g["press"])] = g["action"]
        self.long_press_ms = gesture_map["meta"]["long_press_ms"]

    def resolve(self, button: str, press: str = "short") -> str:
        try:
            return self.table[(button, press)]
        except KeyError:
            raise ValueError(f"no gesture bound for {button} ({press})")
