"""Package the original Real2Gym scene with relative asset paths."""
import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "assets").mkdir(exist_ok=True)
    root = ET.parse(args.source).getroot()
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
        compiler.attrib.pop("texturedir", None)
        compiler.attrib.pop("assetdir", None)
    manifest = []
    for element in root.iter():
        source_text = element.get("file")
        if not source_text:
            continue
        source = Path(source_text)
        if not source.is_absolute():
            source = args.source.parent / source
        data = source.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        relative = Path("assets") / f"{digest[:16]}{source.suffix.lower()}"
        target = args.out / relative
        if not target.exists():
            shutil.copyfile(source, target)
        element.set("file", relative.as_posix())
        manifest.append({"name": element.get("name"), "source_name": source.name,
                         "path": relative.as_posix(), "sha256": digest})
    ET.ElementTree(root).write(args.out / "scene.xml", encoding="unicode")
    (args.out / "asset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Packaged {len(manifest)} asset references into {len(set(x['path'] for x in manifest))} files")


if __name__ == "__main__":
    main()
