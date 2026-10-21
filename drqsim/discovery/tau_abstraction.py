# -*- coding: utf-8 -*-
"""Observable Petri gateway abstraction.

This module separates Petri structural conflicts from business-visible choices.
Silent transitions are explored as routing paths, but the concrete Petri
transitions are retained for execution.
"""

from collections import deque


def transition_key(t):
    if t is None:
        return ""
    name = getattr(t, "name", None)
    return str(name) if name is not None else str(t)


def visible_successors(start_transition, max_depth=20):
    """Return visible labels reachable through tau closure."""
    if start_transition is None:
        return set()

    result = set()
    queue = deque([(start_transition, 0)])
    visited = set()

    while queue:
        transition, depth = queue.popleft()
        key = transition_key(transition)

        if (key, depth) in visited:
            continue
        visited.add((key, depth))

        if depth > max_depth:
            continue

        label = getattr(transition, "label", None)
        if label:
            result.add(str(label))
            continue

        for arc in getattr(transition, "out_arcs", []) or []:
            place = getattr(arc, "target", None)
            if place is None:
                continue

            for arc2 in getattr(place, "out_arcs", []) or []:
                nxt = getattr(arc2, "target", None)
                if nxt is not None:
                    queue.append((nxt, depth + 1))

    return result


def build_observable_gateway(place, max_depth=20):
    """Convert a Petri conflict place into an observable gateway."""
    branches = {}

    for arc in getattr(place, "out_arcs", []) or []:
        transition = getattr(arc, "target", None)
        if transition is None:
            continue

        key = transition_key(transition)
        labels = sorted(visible_successors(transition, max_depth))

        branches[key] = {
            "transition": transition,
            "labels": labels,
            "observable": len(labels) > 0,
        }

    return {
        "source_place": transition_key(place),
        "branches": branches,
        "observable_branches": {
            k: v["labels"][0]
            for k, v in branches.items()
            if v["observable"] and len(v["labels"]) == 1
        },
        "unresolved": [
            k for k, v in branches.items()
            if not v["observable"]
        ],
    }
