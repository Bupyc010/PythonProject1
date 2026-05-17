import math

def center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def is_below(a, b, tolerance=15):
    return b["bbox"][1] >= a["bbox"][3] - tolerance

def horizontal_distance(a, b):
    ax, _ = center(a["bbox"])
    bx, _ = center(b["bbox"])
    return abs(ax - bx)

def vertical_distance(a, b):
    return b["bbox"][1] - a["bbox"][3]

def distance(a, b):
    ax, ay = center(a["bbox"])
    bx, by = center(b["bbox"])
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

def find_connections(elements):
    connections = []

    for i, el in enumerate(elements):
        label = el["label"]

        if label == "terminal" and "конец" in el.get("text", "").lower():
            continue

        candidates = []
        for j, target in enumerate(elements):
            if i == j:
                continue
            if is_below(el, target):
                candidates.append((j, distance(el, target), horizontal_distance(el, target)))

        candidates.sort(key=lambda x: (x[1], x[2]))

        if label == "decision":
            for item in candidates[:2]:
                connections.append((i, item[0]))
        else:
            if candidates:
                connections.append((i, candidates[0][0]))

    return connections