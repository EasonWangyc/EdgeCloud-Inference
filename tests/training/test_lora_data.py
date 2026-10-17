"""LoRA 弱监督数据选择的无模型测试。"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock

from scripts.data.generate_lora_dataset import (
    normalize_json_fences,
    select_group_disjoint_images,
    source_group_id,
)


class LoraDatasetSelectionTests(unittest.TestCase):
    def test_selects_one_image_per_source_group_with_disjoint_splits(self) -> None:
        image_root = Mock()
        image_root.glob.return_value = [
            Path(f"p2_img{group_index}_{frame_index:04d}.jpg")
            for group_index in range(10)
            for frame_index in range(2)
        ]
        selected = select_group_disjoint_images(
            image_root,
            train_count=6,
            validation_count=2,
            seed=42,
        )

        train_groups = {group for _, group, split in selected if split == "train"}
        validation_groups = {
            group for _, group, split in selected if split == "validation"
        }
        self.assertEqual(len(selected), 8)
        self.assertEqual(len(train_groups), 6)
        self.assertEqual(len(validation_groups), 2)
        self.assertTrue(train_groups.isdisjoint(validation_groups))
        self.assertEqual(source_group_id(Path("p2_img28_0408.jpg")), "p2_img28")

    def test_normalizes_complete_json_fence_only(self) -> None:
        raw = "```json\n{\"risk_level\": \"low\"}\n```"
        normalized, changed = normalize_json_fences(raw)
        self.assertTrue(changed)
        self.assertEqual(normalized, '{"risk_level": "low"}')

        raw_with_prefix = "model output:\n```json\n{}\n```"
        normalized, changed = normalize_json_fences(raw_with_prefix)
        self.assertFalse(changed)
        self.assertEqual(normalized, raw_with_prefix)


if __name__ == "__main__":
    unittest.main()
