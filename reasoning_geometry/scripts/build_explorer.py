#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from common import IMAGE_MODELS, pca_2d, read_json, read_jsonl, topk_indices, write_json

HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ReasonCLIP Reasoning Geometry Explorer</title>
  <style>
    :root { color-scheme: light; --bg:#f7f8fa; --panel:#ffffff; --ink:#1d2430; --muted:#607086; --line:#d8dde6; --accent:#2474d4; --accent2:#14956f; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--ink); background:var(--bg); }
    header { padding:16px 22px 10px; border-bottom:1px solid var(--line); background:#fff; position:sticky; top:0; z-index:3; }
    h1 { margin:0; font-size:20px; line-height:1.25; font-weight:720; letter-spacing:0; }
    .sub { margin-top:5px; color:var(--muted); font-size:13px; }
    main { display:grid; grid-template-columns:minmax(460px, 1.05fr) minmax(420px, .95fr); gap:14px; padding:14px; max-width:1680px; margin:0 auto; }
    section { background:var(--panel); border:1px solid var(--line); border-radius:8px; min-width:0; }
    .section-head { display:flex; align-items:center; justify-content:space-between; gap:10px; padding:10px 12px; border-bottom:1px solid var(--line); }
    .section-head h2 { margin:0; font-size:14px; font-weight:700; }
    .controls { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
    select, button { height:32px; border:1px solid var(--line); background:#fff; color:var(--ink); border-radius:6px; padding:0 9px; font-size:13px; }
    button { cursor:pointer; }
    button.active { background:var(--accent); color:white; border-color:var(--accent); }
    canvas { width:100%; height:520px; display:block; background:#fbfcfe; }
    .split { display:grid; grid-template-columns:1fr; gap:14px; }
    .metrics { padding:12px; display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:8px; }
    .metric { border:1px solid var(--line); border-radius:8px; padding:8px; }
    .metric .k { color:var(--muted); font-size:11px; }
    .metric .v { font-size:15px; font-weight:700; margin-top:4px; }
    .retrieval { padding:10px 12px 12px; }
    .model-columns { display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:10px; margin-bottom:12px; }
    .model-title { font-size:12px; color:var(--muted); margin:0 0 6px; font-weight:700; }
    .grid { display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:6px; }
    .thumb { border:1px solid var(--line); border-radius:7px; overflow:hidden; background:#eef2f6; min-height:72px; cursor:pointer; }
    .thumb img { width:100%; aspect-ratio:1 / 1; object-fit:cover; display:block; }
    .score { font-size:10px; color:var(--muted); padding:3px 5px 4px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .detail { padding:12px; display:grid; grid-template-columns:180px 1fr; gap:12px; }
    .detail img { width:180px; height:180px; object-fit:cover; border-radius:8px; border:1px solid var(--line); background:#eef2f6; }
    .detail h3 { margin:0 0 6px; font-size:15px; }
    .detail p { margin:0 0 8px; font-size:12px; color:#2d3848; line-height:1.4; }
    .label { color:var(--muted); font-size:11px; font-weight:700; text-transform:uppercase; margin:8px 0 4px; }
    .neighbors { padding:0 12px 12px; }
    .neighbors .grid { grid-template-columns:repeat(6, minmax(0, 1fr)); }
    .full { grid-column:1 / -1; }
    @media (max-width: 980px) { main { grid-template-columns:1fr; } canvas { height:420px; } .model-columns { grid-template-columns:1fr; } .detail { grid-template-columns:1fr; } .detail img { width:100%; height:auto; aspect-ratio:1 / 1; } .neighbors .grid { grid-template-columns:repeat(3, minmax(0, 1fr)); } }
  </style>
</head>
<body>
  <header>
    <h1>ReasonCLIP Reasoning Geometry Explorer</h1>
    <div class="sub" id="summary"></div>
  </header>
  <main>
    <section>
      <div class="section-head">
        <h2>Embedding Space</h2>
        <div class="controls">
          <select id="modelSelect"></select>
          <button id="colorType" class="active">reasoning type</button>
          <button id="colorSplit">split</button>
        </div>
      </div>
      <canvas id="scatter" width="1000" height="640"></canvas>
      <div class="metrics" id="metrics"></div>
    </section>
    <div class="split">
      <section>
        <div class="section-head">
          <h2>Explore What The Models Retrieve</h2>
          <div class="controls"><select id="promptSelect"></select><select id="familySelect"><option value="clip_l14_224">CLIP</option><option value="siglip_so400m_384">SigLIP</option></select></div>
        </div>
        <div class="retrieval" id="retrieval"></div>
      </section>
      <section>
        <div class="section-head"><h2>Selected Image</h2><div class="controls"><span id="selectedId" class="sub"></span></div></div>
        <div class="detail" id="detail"></div>
        <div class="neighbors"><div class="label">nearest image neighbors in selected model</div><div class="grid" id="neighbors"></div></div>
      </section>
    </div>
  </main>
  <script src="data/explorer_data.js"></script>
  <script>
    const data = window.EXPLORER_DATA;
    const records = data.records;
    const models = data.models;
    const coords = data.coords;
    const neighbors = data.neighbors;
    const metrics = data.metrics || {models:{}};
    const retrievals = data.retrievals || {prompts:[], models:{}};
    let selectedModel = Object.keys(models)[0];
    let selectedIndex = 0;
    let colorMode = 'type';
    const typeColors = {support:'#2474d4', containment:'#14956f', protection:'#c77700', use_or_function:'#7567d9', state_or_activity:'#cf3f5d', spatial_relation:'#07859b', material_state:'#8b6f00', affordance:'#7b7f89', other:'#9aa4b2'};
    const splitColors = {train:'#2474d4', test:'#cf3f5d', val:'#14956f', validation:'#14956f', unknown:'#9aa4b2'};

    function imgSrc(path) { return path || ''; }
    function fmt(x) { return Number.isFinite(x) ? x.toFixed(4) : 'n/a'; }
    function byId(id) { return document.getElementById(id); }
    function esc(s) { return String(s || '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

    function init() {
      byId('summary').textContent = `${records.length} images | ${Object.keys(models).length} visual encoders | caption reference: ${data.caption_reference || 'language-only embeddings'}`;
      const ms = byId('modelSelect');
      Object.entries(models).forEach(([key, meta]) => { const o=document.createElement('option'); o.value=key; o.textContent=meta.label || key; ms.appendChild(o); });
      ms.value = selectedModel;
      ms.onchange = () => { selectedModel = ms.value; draw(); renderMetrics(); renderNeighbors(); };
      byId('colorType').onclick = () => { colorMode='type'; byId('colorType').classList.add('active'); byId('colorSplit').classList.remove('active'); draw(); };
      byId('colorSplit').onclick = () => { colorMode='split'; byId('colorSplit').classList.add('active'); byId('colorType').classList.remove('active'); draw(); };
      const ps = byId('promptSelect');
      retrievals.prompts.forEach(p => { const o=document.createElement('option'); o.value=p.id; o.textContent=p.text; ps.appendChild(o); });
      ps.onchange = renderRetrieval;
      byId('familySelect').onchange = renderRetrieval;
      byId('scatter').addEventListener('click', scatterClick);
      draw(); renderMetrics(); renderRetrieval(); selectIndex(0);
    }

    function pointColor(rec) {
      if (colorMode === 'split') return splitColors[rec.split] || splitColors.unknown;
      return typeColors[rec.primary_reasoning_type] || typeColors.other;
    }

    function canvasMapping(c) {
      const xs = c.map(p => p[0]), ys = c.map(p => p[1]);
      const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
      const canvas = byId('scatter');
      const pad = 28;
      return function(p) {
        const x = pad + (p[0]-minX) / Math.max(1e-6, maxX-minX) * (canvas.width - 2*pad);
        const y = canvas.height - pad - (p[1]-minY) / Math.max(1e-6, maxY-minY) * (canvas.height - 2*pad);
        return [x,y];
      };
    }

    function draw() {
      const canvas = byId('scatter');
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0,0,canvas.width,canvas.height);
      const c = coords[selectedModel];
      if (!c) return;
      const map = canvasMapping(c);
      for (let i=0; i<c.length; i++) {
        const [x,y] = map(c[i]);
        ctx.beginPath();
        ctx.fillStyle = pointColor(records[i]);
        ctx.globalAlpha = i === selectedIndex ? 1 : 0.72;
        ctx.arc(x,y,i === selectedIndex ? 5 : 2.3,0,Math.PI*2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    }

    function scatterClick(evt) {
      const canvas = byId('scatter');
      const rect = canvas.getBoundingClientRect();
      const x = (evt.clientX - rect.left) / rect.width * canvas.width;
      const y = (evt.clientY - rect.top) / rect.height * canvas.height;
      const c = coords[selectedModel];
      const map = canvasMapping(c);
      let best = 0, bestD = Infinity;
      for (let i=0; i<c.length; i++) {
        const [px,py] = map(c[i]);
        const d = (px-x)*(px-x) + (py-y)*(py-y);
        if (d < bestD) { bestD = d; best = i; }
      }
      selectIndex(best);
    }

    function renderMetrics() {
      const m = metrics.models && metrics.models[selectedModel] ? metrics.models[selectedModel] : {};
      const keys = ['neighbor_overlap_at_10','image_similarity_of_caption_neighbors_at_10','rsa_spearman_sampled','triplet_margin_caption_pos_vs_image_hard_neg','triplet_positive_image_similarity','triplet_hard_negative_image_similarity'];
      byId('metrics').innerHTML = keys.map(k => `<div class="metric"><div class="k">${k}</div><div class="v">${fmt(m[k])}</div></div>`).join('');
    }

    function thumb(idx, score) {
      const r = records[idx];
      return `<div class="thumb" onclick="selectIndex(${idx})"><img src="${esc(imgSrc(r.image_path))}" loading="lazy"><div class="score">${esc(r.image_id)}${score !== undefined ? ' | '+fmt(score) : ''}</div></div>`;
    }

    function renderRetrieval() {
      const promptId = byId('promptSelect').value || (retrievals.prompts[0] && retrievals.prompts[0].id);
      const family = byId('familySelect').value;
      const stageOrder = ['baseline','s1','s2'];
      const keys = Object.keys(models).filter(k => models[k].family === family).sort((a,b) => stageOrder.indexOf(models[a].stage) - stageOrder.indexOf(models[b].stage));
      byId('retrieval').innerHTML = `<div class="model-columns">` + keys.map(k => {
        const hits = retrievals.models[k] && retrievals.models[k][promptId] ? retrievals.models[k][promptId] : [];
        return `<div><div class="model-title">${esc(models[k].label || k)}</div><div class="grid">${hits.slice(0,9).map(h => thumb(h.index, h.score)).join('')}</div></div>`;
      }).join('') + `</div>`;
    }

    function selectIndex(idx) {
      selectedIndex = idx;
      const r = records[idx];
      byId('selectedId').textContent = r.image_id;
      byId('detail').innerHTML = `<img src="${esc(imgSrc(r.image_path))}"><div><h3>${esc(r.image_id)}</h3><div class="label">reasoning</div><p>${esc(r.reasoning_caption)}</p><div class="label">source caption</div><p>${esc(r.source_caption)}</p></div>`;
      draw(); renderNeighbors();
    }

    function renderNeighbors() {
      const ns = (neighbors[selectedModel] && neighbors[selectedModel][selectedIndex]) || [];
      byId('neighbors').innerHTML = ns.slice(0,12).map(idx => thumb(idx)).join('');
    }

    window.selectIndex = selectIndex;
    init();
  </script>
</body>
</html>
"""


def primary_reasoning(row):
    caps = row.get("reasoning_captions") or []
    if caps and isinstance(caps[0], dict):
        return caps[0].get("caption", ""), caps[0].get("reasoning_type", "other")
    if caps:
        return str(caps[0]), "other"
    return row.get("descriptive_caption") or row.get("source_caption") or "", "other"


def safe_float_pair(pair):
    return [round(float(pair[0]), 6), round(float(pair[1]), 6)]


def build_data(args):
    rows = read_jsonl(args.annotations)
    ids = [str(r["image_id"]) for r in rows]
    emb_dir = Path(args.image_embedding_dir)
    saved_ids = read_json(emb_dir / "ids.json")
    if list(saved_ids) != ids:
        raise ValueError("annotation ids do not match image embedding ids")
    records = []
    for row in rows:
        reasoning, typ = primary_reasoning(row)
        records.append({
            "image_id": str(row["image_id"]),
            "split": row.get("split", "unknown"),
            "image_path": row.get("image_path", ""),
            "source_caption": row.get("source_caption", ""),
            "descriptive_caption": row.get("descriptive_caption", ""),
            "reasoning_caption": reasoning,
            "primary_reasoning_type": typ,
        })

    model_keys = [p.stem for p in sorted(emb_dir.glob("*.npy")) if p.stem in IMAGE_MODELS]
    coords = {}
    neighbors = {}
    model_meta = {}
    for key in model_keys:
        arr = np.load(emb_dir / f"{key}.npy").astype(np.float32)
        coords[key] = [safe_float_pair(p) for p in pca_2d(arr)]
        sim = arr @ arr.T
        nn, _ = topk_indices(sim, k=args.neighbor_k, exclude_self=True)
        neighbors[key] = nn.astype(int).tolist()
        meta_path = emb_dir / f"{key}.meta.json"
        model_meta[key] = read_json(meta_path) if meta_path.exists() else IMAGE_MODELS[key]
        print(f"built explorer coordinates/neighbors for {key}")

    metrics = read_json(args.metrics) if args.metrics else {}
    retrievals = read_json(args.retrievals) if args.retrievals else {"prompts": [], "models": {}}
    caption_reference = None
    cap_meta = Path(args.caption_embeddings).with_name("meta.json")
    if cap_meta.exists():
        meta = read_json(cap_meta)
        caption_reference = meta.get("model") or meta.get("method")
    return {
        "records": records,
        "models": model_meta,
        "coords": coords,
        "neighbors": neighbors,
        "metrics": metrics,
        "retrievals": retrievals,
        "caption_reference": caption_reference,
    }


def main():
    parser = argparse.ArgumentParser(description="Build static explorer files.")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--caption-embeddings", required=True)
    parser.add_argument("--image-embedding-dir", required=True)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--retrievals", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--neighbor-k", type=int, default=12)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data = build_data(args)
    write_json(data_dir / "explorer_data.json", data)
    js = "window.EXPLORER_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n"
    (data_dir / "explorer_data.js").write_text(js, encoding="utf-8")
    (out_dir / "index.html").write_text(HTML, encoding="utf-8")
    print(f"wrote explorer to {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
