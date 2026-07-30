"""Build a lightweight local Chinese knowledge base from CMRC2018 contexts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def split_spans(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", "", text)
    parts = re.split(r"(?<=[。！？；!?;])", cleaned)
    spans = [part.strip() for part in parts if 12 <= len(part.strip()) <= 260]
    return spans or ([cleaned[:260]] if cleaned else [])


def load_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in data:
        title = str(item.get("title", "")).strip()
        context = str(item.get("context_text", item.get("context", ""))).strip()
        for idx, span in enumerate(split_spans(context)):
            rows.append({"id": f"{path.stem}:{len(rows)}:{idx}", "title": title, "text": span})
    return rows


def seed_rows() -> list[dict]:
    seeds = [
        ("香港", "香港是中华人民共和国特别行政区，位于中国南部、珠江口以东，北接广东省深圳市。"),
        ("澳门", "澳门是中华人民共和国特别行政区，位于中国南部、珠江口西侧，毗邻广东省珠海市。"),
        ("北京", "北京是中华人民共和国首都，位于中国华北地区。"),
        ("上海", "上海是中国直辖市，位于中国东部、长江入海口附近。"),
        ("广州", "广州是广东省省会，位于中国南部、珠江三角洲北缘。"),
        ("深圳", "深圳是广东省副省级市，位于中国南部，毗邻香港。"),
        ("台湾", "台湾位于中国东南沿海的大陆架上，西隔台湾海峡与福建省相望。"),
        ("杭州", "杭州是浙江省省会，位于中国东部、钱塘江下游地区。"),
        ("南京", "南京是江苏省省会，位于中国东部、长江下游地区。"),
        ("成都", "成都是四川省省会，位于中国西南地区、四川盆地西部。"),
        ("重庆", "重庆是中国直辖市，位于中国西南地区、长江上游。"),
        ("天津", "天津是中国直辖市，位于中国华北地区、海河下游。"),
        ("中国", "中华人民共和国位于亚洲东部、太平洋西岸，首都是北京。"),
        ("长江", "长江是中国第一大河，发源于青藏高原，最终注入东海。"),
        ("黄河", "黄河是中国第二长河，流经中国北方多个省区，最终注入渤海。"),
        ("太阳系", "太阳系以太阳为中心，包含八大行星以及矮行星、小行星、彗星等天体。"),
        ("地球", "地球是太阳系中的第三颗行星，也是目前已知存在生命的行星。"),
        ("水", "水在标准大气压下通常在零摄氏度结冰，在一百摄氏度沸腾。"),
        ("光合作用", "光合作用是绿色植物利用光能把二氧化碳和水合成为有机物并释放氧气的过程。"),
        ("牛顿", "艾萨克·牛顿是英国科学家，提出了经典力学定律和万有引力定律。"),
        ("爱因斯坦", "阿尔伯特·爱因斯坦提出了狭义相对论和广义相对论。"),
    ]
    return [{"id": f"seed:{i}", "title": title, "text": text} for i, (title, text) in enumerate(seeds)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("external_assets/cmrc2018"))
    parser.add_argument("--output", type=Path, default=Path("external_assets/kb/cmrc2018_zh_kb.jsonl"))
    parser.add_argument("--include-seed", choices=["on", "off"], default="on")
    args = parser.parse_args()
    rows = []
    if args.include_seed == "on":
        rows.extend(seed_rows())
    for name in ["cmrc2018_train.json", "cmrc2018_dev.json"]:
        path = args.data_dir / name
        if path.exists():
            rows.extend(load_rows(path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
