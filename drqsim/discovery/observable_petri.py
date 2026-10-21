# -*- coding: utf-8 -*-
"""Third version observable Petri projection utilities.

Builds a stable mapping from visible branches to concrete Petri transitions.
The decision model chooses visible behavior; execution resolves the hidden path.
"""

from collections import deque

def tkey(t):
    return str(getattr(t, "name", t)) if t is not None else ""

def visible_projection_from_transition(t, max_depth=32):
    q=deque([(t,0)])
    seen=set()
    labels=set()
    concrete=[]
    while q:
        cur,d=q.popleft()
        k=(tkey(cur),d)
        if k in seen or d>max_depth:
            continue
        seen.add(k)
        concrete.append(tkey(cur))
        label=getattr(cur,"label",None)
        if label:
            labels.add(str(label))
            continue
        for a in getattr(cur,"out_arcs",[]) or []:
            p=getattr(a,"target",None)
            for a2 in getattr(p,"out_arcs",[]) or []:
                nxt=getattr(a2,"target",None)
                if nxt is not None:
                    q.append((nxt,d+1))
    return {"labels":sorted(labels),"path":concrete}

def build_observable_projection(net):
    result={}
    for p in getattr(net,"places",[]) or []:
        branches={}
        for a in getattr(p,"out_arcs",[]) or []:
            t=getattr(a,"target",None)
            if t is None: continue
            branches[tkey(t)]=visible_projection_from_transition(t)
        if len(branches)>=2:
            result[tkey(p)]={"place":p,"branches":branches}
    return result
