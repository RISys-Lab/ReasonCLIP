"""Export the project page into the sczhang.com Jekyll repo.

HTML/CSS/JS go to <site>/reasonclip/. Images and data files go to
<site>/local-assets/reasonclip-page/ for upload to R2 (see the site's ASSETS.md),
and the page references them under https://assets.sczhang.com/<prefix>/.

Usage:
    python website/export_sczhang.py --site ../sczhang --prefix reasonclip-page/v1
"""
import argparse
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site", required=True, help="Path to the sczhang.com repository")
    parser.add_argument("--prefix", default="reasonclip-page/v1", help="R2 key prefix; use a new version for changed assets")
    parser.add_argument("--asset-host", default="https://assets.sczhang.com")
    args = parser.parse_args()

    site = Path(args.site).resolve()
    page_dir = site / "reasonclip"
    upload_dir = site / "local-assets" / args.prefix.replace("/", "-")
    base = f"{args.asset_host.rstrip('/')}/{args.prefix.strip('/')}"

    # 1) R2 payload: assets/ and data/, with image paths in bench.js made absolute
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    shutil.copytree(HERE / "assets", upload_dir / "assets")
    (upload_dir / "data").mkdir(parents=True)
    shutil.copy2(HERE / "data" / "results.js", upload_dir / "data" / "results.js")
    bench = (HERE / "data" / "bench.js").read_text(encoding="utf-8")
    bench = bench.replace('"img": "assets/', f'"img": "{base}/assets/')
    (upload_dir / "data" / "bench.js").write_text(bench, encoding="utf-8")

    # 2) Page files in the site repo
    html = (HERE / "index.html").read_text(encoding="utf-8")
    html = re.sub(r'(src|href|content)="(assets|data)/([^"?]+)(\?v=\d+)?"', lambda m: f'{m[1]}="{base}/{m[2]}/{m[3]}"', html)
    page_dir.mkdir(parents=True, exist_ok=True)
    (page_dir / "index.html").write_text(html, encoding="utf-8")
    shutil.copy2(HERE / "style.css", page_dir / "style.css")
    shutil.copy2(HERE / "app.js", page_dir / "app.js")

    left = re.findall(r'(?:src|href)="(?:assets|data)/[^"]*"', html)
    print(f"page   -> {page_dir}")
    print(f"upload -> {upload_dir}  (R2 prefix {args.prefix})")
    print("unrewritten local refs:", left or "none")


if __name__ == "__main__":
    main()
