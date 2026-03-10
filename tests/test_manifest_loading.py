from pathlib import Path

from croma.metrics.pairs import load_manifest


def test_load_manifest_uses_csv_stem_as_dataset_name(tmp_path: Path) -> None:
    manifest_path = tmp_path / "prostate-shift-binary.csv"
    manifest_path.write_text(
        "\n".join(
            [
                "sample_id,image_path,label,scanner_vendor,slide_id,dataset",
                "s0,/tmp/0.png,A,VendorA,sl0,wrong_name",
                "s1,/tmp/1.png,B,VendorB,sl1,wrong_name",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = load_manifest(str(manifest_path), confounder_column="scanner_vendor")

    assert manifest["dataset"].tolist() == [
        "prostate-shift-binary",
        "prostate-shift-binary",
    ]
    assert manifest["confounder"].tolist() == ["VendorA", "VendorB"]
    assert manifest.attrs["confounder_column"] == "scanner_vendor"
