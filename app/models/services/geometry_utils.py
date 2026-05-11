import numpy as np


def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def find_connections(elements):
    """
    elements: список словарей с 'bbox' и 'label'
    Возвращает список кортежей (from_idx, to_idx)
    """
    connections = []
    # Логика упрощена: ищем стрелки и смотрим, что находится у их концов.
    # В идеале здесь нужен алгоритм Хафа или поиск контуров-линий.

    arrows = [e for e in elements if e['label'] == 'arrow']
    blocks = [e for e in elements if e['label'] != 'arrow']

    # Для каждой стрелки найдем ближайший 'выходящий' блок и 'входящий'
    for arrow in arrows:
        ax1, ay1, ax2, ay2 = arrow['bbox']
        a_center = get_center(arrow['bbox'])

        # Упрощенно: считаем, что поток идет сверху вниз или слева направо
        # Находим блок, который выше/левее центра стрелки (предок)
        # И блок, который ниже/правее (потомок)

        source = None
        target = None
        min_dist_s = float('inf')
        min_dist_t = float('inf')

        for b_idx, b in enumerate(elements):
            if b['label'] == 'arrow': continue

            bx, by = get_center(b['bbox'])
            dist = np.sqrt((bx - a_center[0]) ** 2 + (by - a_center[1]) ** 2)

            # Если блок выше стрелки — потенциальный источник
            if by < ay1 + 10 and dist < min_dist_s:
                min_dist_s = dist
                source = b_idx
            # Если блок ниже стрелки — потенциальная цель
            elif by > ay2 - 10 and dist < min_dist_t:
                min_dist_t = dist
                target = b_idx

        if source is not None and target is not None:
            connections.append((source, target))

    return connections