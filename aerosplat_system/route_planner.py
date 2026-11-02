from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


@dataclass
class Edge:
    to: str
    distance_m: float
    risk: float = 0.0
    obstruction: float = 0.0
    flood: float = 0.0
    fire: float = 0.0
    blocked: bool = False

    def cost(self, wr: float = 8.0, wo: float = 10.0, wf: float = 9.0, wfire: float = 12.0) -> float:
        if self.blocked:
            return float("inf")
        return self.distance_m * (1.0 + wr*self.risk + wo*self.obstruction + wf*self.flood + wfire*self.fire)


class RiskAwareAStar:
    """Risk-aware shortest path on a road/node graph."""
    def __init__(self, coordinates: Dict[str, Tuple[float,float]]):
        self.coordinates = coordinates
        self.adj: Dict[str, list[Edge]] = {k: [] for k in coordinates}

    def add_edge(self, a: str, b: str, distance_m: float, **kwargs) -> None:
        self.adj.setdefault(a, []).append(Edge(b, distance_m, **kwargs))
        self.adj.setdefault(b, []).append(Edge(a, distance_m, **kwargs))

    def _heuristic(self, a: str, b: str) -> float:
        ax, ay = self.coordinates[a]; bx, by = self.coordinates[b]
        return math.hypot(ax-bx, ay-by)

    def solve(self, start: str, goal: str, weights: dict | None = None) -> dict:
        if start not in self.adj or goal not in self.adj:
            raise KeyError("Start/goal node not present in graph")
        weights = weights or {}
        pq = [(0.0, start)]
        g = {start: 0.0}
        came = {}
        while pq:
            f, node = heapq.heappop(pq)
            if node == goal:
                break
            if f > g.get(node, float("inf")) + self._heuristic(node, goal) + 1e-9:
                continue
            for edge in self.adj.get(node, []):
                c = edge.cost(weights.get("risk",8.0), weights.get("obstruction",10.0), weights.get("flood",9.0), weights.get("fire",12.0))
                if math.isinf(c):
                    continue
                tentative = g[node] + c
                if tentative < g.get(edge.to, float("inf")):
                    g[edge.to] = tentative
                    came[edge.to] = node
                    heapq.heappush(pq, (tentative+self._heuristic(edge.to, goal), edge.to))
        if goal not in g:
            return {"found": False, "reason": "No feasible route"}
        path = [goal]
        while path[-1] != start:
            path.append(came[path[-1]])
        path.reverse()
        geometric_distance = 0.0
        risk_cost = 0.0
        edge_records = []
        for a,b in zip(path,path[1:]):
            e = next(e for e in self.adj[a] if e.to == b)
            geometric_distance += e.distance_m
            c = e.cost(weights.get("risk",8.0), weights.get("obstruction",10.0), weights.get("flood",9.0), weights.get("fire",12.0))
            risk_cost += c
            edge_records.append({"from":a,"to":b,"distance_m":e.distance_m,"risk":e.risk,"obstruction":e.obstruction,"flood":e.flood,"fire":e.fire,"blocked":e.blocked})
        return {"found": True, "nodes": path, "distance_m": round(geometric_distance,2), "risk_cost": round(risk_cost,2), "edges": edge_records}


def demo_route() -> dict:
    xy = {"S":(0,0),"A":(50,10),"B":(55,-20),"C":(100,0),"D":(150,10),"G":(200,0),"X":(100,40)}
    r = RiskAwareAStar(xy)
    r.add_edge("S","A",51,risk=.10)
    r.add_edge("S","B",58,risk=.25)
    r.add_edge("A","C",52,risk=.90,obstruction=.85)  # dangerous shortcut
    r.add_edge("B","C",49,risk=.12)
    r.add_edge("C","D",50,risk=.08)
    r.add_edge("C","X",45,risk=.70,flood=.60)
    r.add_edge("X","D",55,risk=.55)
    r.add_edge("D","G",50,risk=.06)
    return r.solve("S","G")
