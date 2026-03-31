import json
from pathlib import Path


def parse_layer_head_spec_text(spec: str):
    if not spec:
        return None
    layer_head_map = {}
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(
                f"Invalid layer/head chunk '{chunk}'. Expected format like '0:5,6'"
            )
        layer_str, heads_str = chunk.split(":", 1)
        layer_idx = int(layer_str.strip())
        heads = []
        for item in heads_str.split(","):
            item = item.strip()
            if not item:
                continue
            heads.append(int(item))
        if not heads:
            raise ValueError(f"No heads specified for layer {layer_idx}")
        layer_head_map[layer_idx] = tuple(sorted(set(heads)))
    return layer_head_map or None


def parse_layer_head_spec_file(spec_file: str):
    if not spec_file:
        return None
    path = Path(spec_file)
    if not path.exists():
        raise FileNotFoundError(f"Layer/head spec file not found: {spec_file}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        layer_head_map = {}
        for key, value in data.items():
            layer_idx = int(key)
            if not isinstance(value, (list, tuple)) or not value:
                raise ValueError(f"Layer {key} must map to a non-empty list of heads")
            layer_head_map[layer_idx] = tuple(sorted(set(int(v) for v in value)))
        return layer_head_map or None
    return parse_layer_head_spec_text(path.read_text(encoding="utf-8"))


def resolve_layer_head_map(spec: str = "", spec_file: str = ""):
    map_from_text = parse_layer_head_spec_text(spec) if spec else None
    map_from_file = parse_layer_head_spec_file(spec_file) if spec_file else None
    if map_from_text is not None and map_from_file is not None:
        raise ValueError("Use either layer_head_spec or layer_head_spec_file, not both")
    return map_from_text if map_from_text is not None else map_from_file
