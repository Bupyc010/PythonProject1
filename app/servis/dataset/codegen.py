def sanitize_text(text):
    text = text.strip().replace("\n", " ")
    return " ".join(text.split())

def make_statement(text, idx):
    t = sanitize_text(text).lower()

    if not t:
        return f'print("step_{idx}")'

    if "=" in text and "==" not in text:
        return text

    if t.startswith("ввод ") or t.startswith("input "):
        name = text.split(" ", 1)[-1].strip()
        if not name:
            name = f"var_{idx}"
        return f'{name} = input("{name}: ")'

    if t.startswith("вывод ") or t.startswith("print "):
        msg = text.split(" ", 1)[-1].strip() if " " in text else f"step_{idx}"
        return f'print({msg!r})'

    return f'print({text!r})'

def make_condition(text, idx):
    t = sanitize_text(text)
    if not t:
        return "x > 0"
    low = t.lower()
    if low.startswith("если "):
        t = t[5:].strip()
    return t if t else f"cond_{idx}"


def generate_smart_code(elements, connections):
    # Сортируем элементы сверху вниз по координате Y
    elements = sorted(elements, key=lambda x: x['bbox'][1])

    lines = ["def main():"]
    indent = "    "

    for el in elements:
        label = el['label'].lower()
        text = el.get('text', '').strip()

        if not text:
            continue

        # 1. Если это блок ввода или процесса
        if label == 'rectangle' or label == 'round_rect':
            # Если текст похож на ввод (содержит input или x,y)
            if 'input' in text or '=' in text:
                lines.append(f"{indent}{text}")
            # Если это просто вывод (на твоей схеме 'YES'/'NO')
            elif "'" in text or '"' in text or text.isupper():
                lines.append(f"{indent}print({text})")
            else:
                lines.append(f"{indent}{text}")

        # 2. Если это условие (ромб или шестиугольник)
        elif label == 'diamond' or label == 'hexagon':
            # Чистим текст условия
            condition = text.replace('да', '').replace('нет', '').strip()
            lines.append(f"{indent}if {condition}:")
            indent = "        "  # Увеличиваем отступ для следующего блока (YES)

    # Если в коде нет веток, добавляем заглушку pass
    if len(lines) == 1:
        lines.append(f"{indent}pass")

    lines.append("\nif __name__ == '__main__':")
    lines.append("    main()")

    return "\n".join(lines)