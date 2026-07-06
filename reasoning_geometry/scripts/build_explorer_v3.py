#!/usr/bin/env python3
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageOps

from common import IMAGE_MODELS, first_sentence, l2_normalize, pca_2d, read_json, read_jsonl, topk_indices

HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ReasonCLIP Model-Centric Explorer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f6f8;
      --panel: #ffffff;
      --ink: #18202b;
      --muted: #5d6b7c;
      --line: #d7dde6;
      --soft: #edf1f6;
      --blue: #2368b5;
      --green: #13845f;
      --red: #b9434a;
      --amber: #9b6a00;
      --violet: #6e5fc7;
      --shadow: 0 1px 2px rgba(16, 24, 40, 0.05);
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }
    header { position: sticky; top: 0; z-index: 5; background: #fff; border-bottom: 1px solid var(--line); padding: 12px 16px; }
    h1 { margin: 0; font-size: 19px; line-height: 1.25; letter-spacing: 0; }
    .sub { margin-top: 4px; font-size: 12px; color: var(--muted); }
    .toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-top: 10px; }
    .control-label { font-size: 12px; color: var(--muted); font-weight: 700; }
    label { display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--muted); }
    select, button { height: 32px; border: 1px solid var(--line); background: #fff; color: var(--ink); border-radius: 6px; padding: 0 9px; font-size: 13px; letter-spacing: 0; }
    button { cursor: pointer; }
    button.active { border-color: var(--blue); background: var(--blue); color: #fff; }
    .segmented { display: inline-flex; align-items: center; gap: 2px; padding: 2px; border: 1px solid var(--line); border-radius: 7px; background: var(--soft); }
    .segmented button { height: 28px; border-color: transparent; background: transparent; padding: 0 10px; font-size: 12px; }
    .segmented button.active { border-color: var(--blue); background: var(--blue); color: #fff; }
    .segmented.compact button { padding: 0 8px; }
    .section-control select { height: 28px; }
    main { display: grid; grid-template-columns: minmax(430px, 0.9fr) minmax(620px, 1.1fr); gap: 12px; padding: 12px; max-width: 1780px; margin: 0 auto; }
    section { min-width: 0; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; box-shadow: var(--shadow); }
    .section-head { min-height: 42px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; padding: 9px 11px; border-bottom: 1px solid var(--line); }
    .section-head h2 { margin: 0; font-size: 13px; line-height: 1.25; letter-spacing: 0; }
    .controls { display: flex; align-items: center; gap: 7px; flex-wrap: wrap; }
    .tab-buttons button { height: 28px; padding: 0 8px; font-size: 12px; }
    .tab-panel.hidden { display: none; }
    .hidden-inline { display: none !important; }
    .stage-control select { height: 28px; max-width: 260px; }
    .query-strip { display: flex; align-items: baseline; flex-wrap: wrap; gap: 8px; padding: 10px 10px 0; font-size: 12px; color: var(--muted); }
    .query-strip strong { color: var(--ink); font-size: 13px; }
    .stack { display: grid; grid-template-columns: 1fr; gap: 12px; }
    .left-stack { position: sticky; top: 96px; align-self: start; max-height: calc(100vh - 108px); overflow: auto; scrollbar-gutter: stable; }
    canvas { width: 100%; height: 480px; display: block; background: #fbfcfe; }
    .selected { display: grid; grid-template-columns: 120px minmax(0, 1fr); gap: 10px; padding: 10px; border-top: 1px solid var(--line); }
    .selected img { width: 120px; height: 120px; object-fit: cover; border-radius: 6px; border: 1px solid var(--line); background: var(--soft); }
    .selected h3 { margin: 0 0 5px; font-size: 14px; }
    .caption { margin: 0; font-size: 12px; line-height: 1.35; color: #2b3645; }
    .columns { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; padding: 10px; }
    .model-column { position: relative; min-width: 0; border: 1px solid var(--line); border-radius: 8px; padding: 8px; background: #fff; overflow: hidden; }
    .model-column::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--muted); }
    .model-column.stage-baseline { background: linear-gradient(180deg, #fafbfd 0%, #fff 52%); border-color: #cfd6e1; }
    .model-column.stage-baseline::before { background: #697789; }
    .model-column.stage-s1 { background: linear-gradient(180deg, #f0f8f4 0%, #fff 52%); border-color: #c5e4d3; }
    .model-column.stage-s1::before { background: var(--green); }
    .model-column.stage-s2 { background: linear-gradient(180deg, #f4f1fb 0%, #fff 52%); border-color: #d8d1ef; }
    .model-column.stage-s2::before { background: var(--violet); }
    .model-column .result-grid { padding-left: 2px; }
    .column-title { display: flex; align-items: center; justify-content: space-between; gap: 6px; margin: 0 0 8px; min-height: 24px; padding: 0 0 6px 7px; border-bottom: 1px solid rgba(150, 160, 175, 0.28); font-size: 12px; color: var(--muted); font-weight: 700; }
    .column-title .stage-name { color: var(--ink); font-size: 12px; }
    .column-title .model-name { text-align: right; font-size: 10px; line-height: 1.15; }
    .result-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 7px; }
    .result-card { min-width: 0; border: 1px solid var(--line); border-radius: 7px; overflow: hidden; background: #fff; cursor: pointer; }
    .result-card.selected-card { border-color: var(--blue); box-shadow: 0 0 0 2px rgba(35, 104, 181, 0.15); }
    .result-card img { width: 100%; aspect-ratio: 1 / 1; object-fit: cover; display: block; background: var(--soft); }
    .meta { padding: 6px; display: grid; gap: 4px; }
    .rank-line { display: flex; align-items: center; justify-content: space-between; gap: 5px; min-height: 18px; font-size: 11px; font-weight: 700; color: #263141; white-space: nowrap; }
    .sim { color: var(--blue); }
    .id-line { min-width: 0; font-size: 10px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .snippet { font-size: 10px; line-height: 1.25; color: #394658; min-height: 38px; max-height: 50px; overflow: hidden; }
    .badge-row { display: flex; align-items: center; flex-wrap: wrap; gap: 4px; min-height: 18px; }
    .badge { display: inline-flex; align-items: center; height: 17px; padding: 0 5px; border-radius: 4px; font-size: 10px; font-weight: 700; background: var(--soft); color: var(--muted); white-space: nowrap; }
    .badge.new { background: #eaf6f0; color: var(--green); }
    .badge.shared { background: #eef3fb; color: var(--blue); }
    .badge.pushed { background: #fbeeee; color: var(--red); }
    .badge.stable { background: #f4f1fb; color: var(--violet); }
    .badge.delta { background: #fff4da; color: var(--amber); }
    .changed { padding: 10px; display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 9px; }
    .changed .result-grid { grid-template-columns: 1fr; }
    .metric-line { font-size: 10px; line-height: 1.25; color: #263141; white-space: normal; }
    .empty { padding: 16px; color: var(--muted); font-size: 12px; }
    @media (max-width: 1180px) { main { grid-template-columns: 1fr; } .left-stack { position: static; max-height: none; overflow: visible; } canvas { height: 420px; } }
    @media (max-width: 760px) { .columns, .changed { grid-template-columns: 1fr; } .result-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); } .selected { grid-template-columns: 1fr; } .selected img { width: 100%; height: auto; aspect-ratio: 1 / 1; } }
    @media (max-width: 520px) { .result-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } header { padding: 10px; } main { padding: 8px; } }
  </style>
</head>
<body>
  <header>
    <h1>ReasonCLIP Model-Centric Explorer</h1>
    <div class="sub" id="summary"></div>
    <div class="toolbar">
      <span class="control-label">encoder</span>
      <div class="segmented" id="encoderButtons"></div>
    </div>
  </header>
  <main>
    <div class="stack left-stack">
      <section>
        <div class="section-head">
          <h2>Embedding Space</h2>
          <div class="controls">
            <span class="sub" id="scatterLabel"></span>
            <label class="section-control">stage <select id="scatterStageSelect"></select></label>
            <span class="control-label">color</span>
            <div class="segmented compact">
              <button id="colorSplit" class="active">dataset split</button>
              <button id="colorAnchor">anchor neighbors</button>
            </div>
          </div>
        </div>
        <canvas id="scatter" width="1000" height="620"></canvas>
        <div class="selected" id="selectedPanel"></div>
      </section>
    </div>
    <div class="stack">
      <section>
        <div class="section-head">
          <h2 id="rightPanelTitle">Inference Retrieval</h2>
          <div class="controls tab-buttons">
            <button id="tabRetrieval" class="active">Retrieval</button>
            <button id="tabNeighbors">Neighbors</button>
            <button id="tabChanged">Changed</button>
            <label id="categoryControl" class="stage-control">category <select id="categorySelect"></select></label>
            <label id="conceptControl" class="stage-control">query <select id="conceptSelect"></select></label>
            <span class="sub hidden-inline" id="anchorTitle"></span>
            <label id="changeStageControl" class="stage-control hidden-inline">compare <select id="changeStageSelect"><option value="s1">S1 vs baseline</option><option value="s2">S2 vs baseline</option></select></label>
          </div>
        </div>
        <div id="retrievalPanel" class="tab-panel"></div>
        <div id="neighborPanel" class="tab-panel hidden"></div>
        <div id="changedPanel" class="tab-panel hidden"></div>
      </section>
    </div>
  </main>
  <script src="data/explorer_core.js"></script>
  <script src="data/explorer_records.js"></script>
  <script src="data/explorer_geometry.js"></script>
  <script src="data/explorer_neighbor_changes.js"></script>
  <script>
    const data = window.EXPLORER_DATA;
    const records = data.records || [];
    const models = data.models || {};
    const families = data.families || {};
    const concepts = data.concepts || [];
    const categories = data.categories || inferCategories(concepts);
    const coords = data.coords || {};
    const neighbors = data.neighbors || {};
    const retrievals = data.retrievals || {models:{}};
    const changes = data.neighbor_changes || {};
    let selectedFamily = Object.keys(families)[0];
    let selectedCategory = categories[0] ? categories[0].id : ((concepts[0] && concepts[0].category) || 'uncategorized');
    let selectedConcept = (firstConceptForCategory(selectedCategory) || concepts[0] || {id:''}).id;
    let selectedIndex = 0;
    let scatterStage = 'baseline';
    let scatterModel = '';
    let colorMode = 'split';
    let activePanel = 'retrieval';
    const splitColors = {train:'#2368b5', test:'#b9434a', val:'#13845f', validation:'#13845f', unknown:'#8c98a8'};

    function byId(id) { return document.getElementById(id); }
    function esc(s) { return String(s || '').replace(/[&<>\"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}[c])); }
    function fmt(x) { return Number.isFinite(Number(x)) ? Number(x).toFixed(4) : 'n/a'; }
    function rankFmt(rank) { return rank && Number.isFinite(Number(rank)) ? '#' + String(rank).padStart(2, '0') : '#--'; }
    function familyKeys() {
      const f = families[selectedFamily] || {};
      return ['baseline','s1','s2'].map(stage => f.keys && f.keys[stage]).filter(Boolean);
    }
    function stageLabel(key) { return (models[key] && (models[key].stage || key)) || key; }
    function stageClass(key) { return (models[key] && (models[key].stage || 'model')) || 'model'; }
    function modelTitle(key) { return (models[key] && (models[key].label || key)) || key; }
    function inferCategories(items) {
      const out = [];
      const seen = new Set();
      items.forEach(c => {
        const id = c.category || 'uncategorized';
        if (seen.has(id)) return;
        seen.add(id);
        out.push({id, label: c.category_label || id});
      });
      return out.length ? out : [{id:'uncategorized', label:'Uncategorized'}];
    }
    function currentCategory() { return categories.find(c => c.id === selectedCategory) || categories[0] || {id:'uncategorized', label:'Uncategorized'}; }
    function conceptsForCategory() { return concepts.filter(c => (c.category || 'uncategorized') === selectedCategory); }
    function firstConceptForCategory(category) { return concepts.find(c => (c.category || 'uncategorized') === category); }
    function currentConcept() { return concepts.find(c => c.id === selectedConcept) || firstConceptForCategory(selectedCategory) || concepts[0] || {id:'', display_label:'query'}; }
    function imgTag(idx) { return `<img src="${esc(records[idx].image_path)}" loading="lazy" alt="">`; }
    function snippet(idx) { return esc(records[idx].caption_snippet || records[idx].descriptive_caption || records[idx].source_caption || ''); }

    function init() {
      byId('summary').textContent = `${records.length} images | ${Object.keys(models).length} visual encoders | inference-only model-centric view`;
      renderEncoderButtons();

      const categorySelect = byId('categorySelect');
      categories.forEach(c => { const o = document.createElement('option'); o.value = c.id; o.textContent = c.label || c.id; categorySelect.appendChild(o); });
      categorySelect.value = selectedCategory;
      categorySelect.onchange = () => {
        selectedCategory = categorySelect.value;
        const first = firstConceptForCategory(selectedCategory);
        selectedConcept = first ? first.id : '';
        syncConceptSelect();
        renderRetrieval();
      };

      const cs = byId('conceptSelect');
      cs.onchange = () => { selectedConcept = cs.value; renderRetrieval(); };
      syncConceptSelect();

      syncScatterStageSelect();
      byId('scatterStageSelect').onchange = () => { scatterStage = byId('scatterStageSelect').value; syncScatterModelFromStage(); renderScatter(); };

      byId('colorSplit').onclick = () => { colorMode = 'split'; byId('colorSplit').classList.add('active'); byId('colorAnchor').classList.remove('active'); renderScatter(); };
      byId('colorAnchor').onclick = () => { colorMode = 'anchor'; byId('colorAnchor').classList.add('active'); byId('colorSplit').classList.remove('active'); renderScatter(); };
      byId('tabRetrieval').onclick = () => setActivePanel('retrieval');
      byId('tabNeighbors').onclick = () => setActivePanel('neighbors');
      byId('tabChanged').onclick = () => setActivePanel('changed');
      byId('changeStageSelect').onchange = renderChanged;
      byId('scatter').addEventListener('click', scatterClick);
      renderAll();
    }

    function stageDisplay(stage) {
      return stage === 'baseline' ? 'baseline' : stage.toUpperCase();
    }

    function availableStages() {
      const keys = ((families[selectedFamily] || {}).keys) || {};
      return ['baseline', 's1', 's2'].filter(stage => keys[stage]);
    }

    function modelKeyForStage(stage) {
      const keys = ((families[selectedFamily] || {}).keys) || {};
      return keys[stage] || keys.baseline || familyKeys()[0] || Object.keys(models)[0] || '';
    }

    function renderEncoderButtons() {
      const wrap = byId('encoderButtons');
      wrap.innerHTML = '';
      Object.entries(families).forEach(([key, meta]) => {
        const b = document.createElement('button');
        b.type = 'button';
        b.textContent = meta.label || key;
        b.className = key === selectedFamily ? 'active' : '';
        b.onclick = () => setFamily(key);
        wrap.appendChild(b);
      });
    }

    function setFamily(key) {
      if (!families[key] || selectedFamily === key) return;
      selectedFamily = key;
      renderEncoderButtons();
      syncScatterStageSelect();
      renderAll();
    }

    function syncScatterStageSelect() {
      const select = byId('scatterStageSelect');
      const stages = availableStages();
      if (!stages.includes(scatterStage)) scatterStage = stages[0] || 'baseline';
      select.innerHTML = '';
      stages.forEach(stage => {
        const o = document.createElement('option');
        o.value = stage;
        o.textContent = stageDisplay(stage);
        select.appendChild(o);
      });
      select.value = scatterStage;
      syncScatterModelFromStage();
    }

    function syncScatterModelFromStage() {
      scatterModel = modelKeyForStage(scatterStage);
    }

    function syncConceptSelect() {
      const cs = byId('conceptSelect');
      const list = conceptsForCategory();
      if (!list.some(c => c.id === selectedConcept)) selectedConcept = list[0] ? list[0].id : '';
      cs.innerHTML = '';
      list.forEach(c => {
        const o = document.createElement('option');
        o.value = c.id;
        o.textContent = c.display_label;
        cs.appendChild(o);
      });
      cs.value = selectedConcept;
    }

    function setActivePanel(panel) {
      activePanel = panel;
      renderActivePanel();
    }

    function renderActivePanel() {
      const cfg = {
        retrieval: ['Inference Retrieval', 'tabRetrieval', 'retrievalPanel'],
        neighbors: ['Anchor Neighbor Compare', 'tabNeighbors', 'neighborPanel'],
        changed: ['Neighbor Changed', 'tabChanged', 'changedPanel']
      };
      Object.entries(cfg).forEach(([key, item]) => {
        byId(item[1]).classList.toggle('active', activePanel === key);
        byId(item[2]).classList.toggle('hidden', activePanel !== key);
      });
      byId('rightPanelTitle').textContent = cfg[activePanel][0];
      byId('categoryControl').classList.toggle('hidden-inline', activePanel !== 'retrieval');
      byId('conceptControl').classList.toggle('hidden-inline', activePanel !== 'retrieval');
      byId('anchorTitle').classList.toggle('hidden-inline', activePanel !== 'neighbors');
      byId('changeStageControl').classList.toggle('hidden-inline', activePanel !== 'changed');
    }

    function renderAll() { renderScatter(); renderSelected(); renderRetrieval(); renderNeighbors(); renderChanged(); renderActivePanel(); }

    function mapPoints(c) {
      const xs = c.map(p => p[0]), ys = c.map(p => p[1]);
      const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
      const canvas = byId('scatter');
      const pad = 26;
      return function(p) {
        const x = pad + (p[0] - minX) / Math.max(1e-6, maxX - minX) * (canvas.width - 2 * pad);
        const y = canvas.height - pad - (p[1] - minY) / Math.max(1e-6, maxY - minY) * (canvas.height - 2 * pad);
        return [x, y];
      };
    }

    function anchorSetForModel(key) {
      const list = (neighbors[key] && neighbors[key][selectedIndex]) || [];
      return new Set(list.slice(0, 18).map(v => v[0]));
    }

    function pointColor(idx) {
      if (idx === selectedIndex) return '#111827';
      if (colorMode === 'anchor') {
        const set = anchorSetForModel(scatterModel);
        return set.has(idx) ? '#d47c00' : '#c2cad5';
      }
      const rec = records[idx] || {};
      return splitColors[rec.split] || splitColors.unknown;
    }

    function renderScatter() {
      const canvas = byId('scatter');
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const c = coords[scatterModel] || [];
      byId('scatterLabel').textContent = modelTitle(scatterModel);
      if (!c.length) return;
      const map = mapPoints(c);
      for (let i = 0; i < c.length; i++) {
        const [x, y] = map(c[i]);
        ctx.beginPath();
        ctx.fillStyle = pointColor(i);
        ctx.globalAlpha = i === selectedIndex ? 1 : 0.74;
        ctx.arc(x, y, i === selectedIndex ? 5.2 : 2.2, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    }

    function scatterClick(evt) {
      const canvas = byId('scatter');
      const rect = canvas.getBoundingClientRect();
      const x = (evt.clientX - rect.left) / rect.width * canvas.width;
      const y = (evt.clientY - rect.top) / rect.height * canvas.height;
      const c = coords[scatterModel] || [];
      const map = mapPoints(c);
      let best = 0, bestD = Infinity;
      for (let i = 0; i < c.length; i++) {
        const [px, py] = map(c[i]);
        const d = (px - x) * (px - x) + (py - y) * (py - y);
        if (d < bestD) { bestD = d; best = i; }
      }
      selectIndex(best);
    }

    function selectIndex(idx) { selectedIndex = idx; renderSelected(); renderScatter(); renderNeighbors(); renderChanged(); }

    function renderSelected() {
      const r = records[selectedIndex] || {};
      byId('selectedPanel').innerHTML = `${imgTag(selectedIndex)}<div><h3>${esc(r.image_id)}</h3><p class="caption">${esc(r.source_caption || r.descriptive_caption || '')}</p></div>`;
      byId('anchorTitle').textContent = r.image_id || '';
    }

    function hitCard(idx, opts) {
      opts = opts || {};
      const r = records[idx] || {};
      const selected = idx === selectedIndex ? ' selected-card' : '';
      const badges = [];
      if (opts.status) badges.push(`<span class="badge ${esc(opts.statusClass || '')}">${esc(opts.status)}</span>`);
      if (opts.deltaText) badges.push(`<span class="badge delta">${esc(opts.deltaText)}</span>`);
      const second = opts.secondLine ? `<div class="metric-line">${opts.secondLine}</div>` : '';
      return `<div class="result-card${selected}" onclick="selectIndex(${idx})">${imgTag(idx)}<div class="meta"><div class="rank-line"><span>${esc(opts.rankText || '')}</span><span class="sim">${esc(opts.simText || '')}</span></div>${second}<div class="id-line">${esc(r.image_id || '')}</div><div class="snippet">${snippet(idx)}</div><div class="badge-row">${badges.join('')}</div></div></div>`;
    }

    function renderRetrieval() {
      const concept = currentConcept();
      const category = currentCategory();
      const keys = familyKeys();
      byId('retrievalPanel').innerHTML = `<div class="query-strip"><span>${esc(category.label || category.id)}</span><strong>${esc(concept.display_label)}</strong></div><div class="columns">${keys.map(key => {
        const hits = (((retrievals.models || {})[key] || {})[concept.id] || []).slice(0, 18);
        const cards = hits.map(h => {
          const status = h.status_vs_baseline === 'new' ? 'new' : (h.status_vs_baseline === 'shared' ? 'shared' : 'baseline');
          const cls = status === 'new' ? 'new' : (status === 'shared' ? 'shared' : '');
          const delta = key.includes('_base') ? '' : `rank ${h.rank_delta_vs_baseline >= 0 ? '+' : ''}${h.rank_delta_vs_baseline}`;
          const second = key.includes('_base') ? '' : `base ${rankFmt(h.baseline_rank)} sim ${fmt(h.baseline_similarity)}`;
          return hitCard(h.index, {rankText: rankFmt(h.rank), simText: `sim ${fmt(h.similarity)}`, status, statusClass: cls, deltaText: delta, secondLine: second});
        }).join('');
        return `<div class="model-column stage-${esc(stageClass(key))}"><div class="column-title"><span class="stage-name">${esc(stageLabel(key).toUpperCase())}</span><span class="model-name">${esc(modelTitle(key))}</span></div><div class="result-grid">${cards}</div></div>`;
      }).join('')}</div>`;
    }

    function renderNeighbors() {
      const keys = familyKeys();
      const baseKey = keys[0];
      const baseSet = new Set(((neighbors[baseKey] || [])[selectedIndex] || []).slice(0, 12).map(v => v[0]));
      byId('neighborPanel').innerHTML = `<div class="columns">${keys.map(key => {
        const list = ((neighbors[key] || [])[selectedIndex] || []).slice(0, 12);
        const cards = list.map((v, i) => {
          const idx = v[0], sim = v[1];
          const status = key === baseKey ? 'baseline' : (baseSet.has(idx) ? 'shared' : 'new');
          const cls = status === 'new' ? 'new' : (status === 'shared' ? 'shared' : '');
          return hitCard(idx, {rankText: rankFmt(i + 1), simText: `sim ${fmt(sim)}`, status, statusClass: cls});
        }).join('');
        return `<div class="model-column stage-${esc(stageClass(key))}"><div class="column-title"><span class="stage-name">${esc(stageLabel(key).toUpperCase())}</span><span class="model-name">${esc(modelTitle(key))}</span></div><div class="result-grid">${cards}</div></div>`;
      }).join('')}</div>`;
    }

    function changeCard(entry, type) {
      const idx = entry[0], rb = entry[1], rs = entry[2], sb = entry[3], ss = entry[4];
      const deltaRank = rb - rs;
      const deltaSim = ss - sb;
      const cls = type === 'pulled' ? 'new' : (type === 'pushed' ? 'pushed' : 'stable');
      const label = type === 'pulled' ? 'pulled' : (type === 'pushed' ? 'pushed' : 'stable');
      const second = `base ${rankFmt(rb)} sim ${fmt(sb)} | stage ${rankFmt(rs)} sim ${fmt(ss)}`;
      return hitCard(idx, {rankText: `${rankFmt(rb)} -> ${rankFmt(rs)}`, simText: `d ${deltaSim >= 0 ? '+' : ''}${fmt(deltaSim)}`, status: label, statusClass: cls, deltaText: `rank ${deltaRank >= 0 ? '+' : ''}${deltaRank}`, secondLine: second});
    }

    function renderChanged() {
      const stage = byId('changeStageSelect').value;
      const familyObj = (changes[selectedFamily] || {})[stage] || {};
      const groups = [
        ['pulled', 'Pulled closer'],
        ['pushed', 'Pushed away'],
        ['stable', 'Stable shared']
      ];
      byId('changedPanel').innerHTML = `<div class="changed">${groups.map(([key, title]) => {
        const rows = (familyObj[key] || [])[selectedIndex] || [];
        const cards = rows.slice(0, 8).map(e => changeCard(e, key)).join('') || '<div class="empty">No entries</div>';
        return `<div><div class="column-title"><span>${title}</span><span>${stage.toUpperCase()}</span></div><div class="result-grid">${cards}</div></div>`;
      }).join('')}</div>`;
    }

    window.selectIndex = selectIndex;
    init();
  </script>
</body>
</html>
"""


def round_float(x: float) -> float:
    return round(float(x), 6)


def safe_pair(pair: np.ndarray) -> List[float]:
    return [round_float(float(pair[0])), round_float(float(pair[1]))]


def family_map(model_keys: List[str]) -> Dict[str, Dict[str, Any]]:
    families: Dict[str, Dict[str, Any]] = {}
    labels = {
        "clip_l14_224": "CLIP-L/14-224",
        "siglip_so400m_384": "SigLIP-So400M/14-384",
    }
    for key in model_keys:
        cfg = IMAGE_MODELS[key]
        family = cfg["family"]
        families.setdefault(family, {"label": labels.get(family, family), "keys": {}})
        families[family]["keys"][cfg["stage"]] = key
    return families


def make_thumb(src: str, dst: Path, size: int, quality: int) -> bool:
    if dst.exists():
        return True
    try:
        with Image.open(src) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            img.thumbnail((size, size), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (size, size), (238, 242, 246))
            x = (size - img.width) // 2
            y = (size - img.height) // 2
            canvas.paste(img, (x, y))
            dst.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(dst, format="JPEG", quality=quality, optimize=True)
        return True
    except Exception as exc:
        print(f"warning: failed to make thumbnail for {src}: {exc}")
        return False


def build_records(rows: List[Dict[str, Any]], out_dir: Path, thumb_size: int, thumb_quality: int, make_thumbs: bool) -> List[Dict[str, Any]]:
    records = []
    thumb_dir = out_dir / "assets" / "thumbs"
    missing = 0
    for row in rows:
        image_id = str(row["image_id"])
        thumb_rel = f"assets/thumbs/{image_id}.jpg"
        src = str(row.get("image_path") or "")
        if make_thumbs:
            ok = make_thumb(src, thumb_dir / f"{image_id}.jpg", thumb_size, thumb_quality)
            if not ok:
                missing += 1
        caption = str(row.get("source_caption") or row.get("descriptive_caption") or "")
        records.append({
            "image_id": image_id,
            "split": row.get("split", "unknown"),
            "image_path": thumb_rel,
            "source_caption": caption,
            "descriptive_caption": row.get("descriptive_caption", ""),
            "caption_snippet": first_sentence(caption, max_words=34),
        })
    if missing:
        print(f"warning: {missing} thumbnails failed")
    return records


def rank_matrix(sim: np.ndarray) -> np.ndarray:
    order = np.argsort(-sim, axis=1)
    ranks = np.empty(order.shape, dtype=np.uint16)
    values = np.arange(1, sim.shape[1] + 1, dtype=np.uint16)
    rows = np.arange(sim.shape[0])[:, None]
    ranks[rows, order] = values[None, :]
    return ranks


def entry(idx: int, rb: int, rs: int, sb: float, ss: float) -> List[Any]:
    return [int(idx), int(rb), int(rs), round_float(sb), round_float(ss)]


def compute_neighbors_and_changes(emb_dir: Path, model_keys: List[str], neighbor_k: int, change_pool: int, change_k: int, compare_k: int) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    coords: Dict[str, Any] = {}
    neighbors: Dict[str, Any] = {}
    changes: Dict[str, Any] = {}
    families = family_map(model_keys)

    for key in model_keys:
        arr = np.load(emb_dir / f"{key}.npy").astype(np.float32)
        arr = l2_normalize(arr)
        coords[key] = [safe_pair(p) for p in pca_2d(arr)]
        sim = arr @ arr.T
        np.fill_diagonal(sim, -np.inf)
        nn, ss = topk_indices(sim, k=neighbor_k, exclude_self=False)
        neighbors[key] = [[[int(idx), round_float(score)] for idx, score in zip(row_i, row_s)] for row_i, row_s in zip(nn, ss)]
        print(f"built coords/neighbors for {key}")

    for family, meta in families.items():
        keys = meta["keys"]
        base_key = keys.get("baseline")
        if not base_key:
            continue
        family_embs = {}
        family_sims = {}
        family_ranks = {}
        family_top = {}
        family_scores = {}
        for key in [keys.get("baseline"), keys.get("s1"), keys.get("s2")]:
            if not key:
                continue
            arr = l2_normalize(np.load(emb_dir / f"{key}.npy").astype(np.float32))
            sim = arr @ arr.T
            np.fill_diagonal(sim, -np.inf)
            family_embs[key] = arr
            family_sims[key] = sim
            family_ranks[key] = rank_matrix(sim)
            top, top_scores = topk_indices(sim, k=change_pool, exclude_self=False)
            family_top[key] = top
            family_scores[key] = top_scores
            print(f"prepared change ranks for {key}")

        base_top = family_top[base_key]
        base_sim = family_sims[base_key]
        base_ranks = family_ranks[base_key]
        changes[family] = {}
        for stage in ["s1", "s2"]:
            stage_key = keys.get(stage)
            if not stage_key:
                continue
            stage_top = family_top[stage_key]
            stage_sim = family_sims[stage_key]
            stage_ranks = family_ranks[stage_key]
            pulled_all = []
            pushed_all = []
            stable_all = []
            n = base_top.shape[0]
            for i in range(n):
                base_compare = set(int(x) for x in base_top[i, :compare_k])
                stage_compare = set(int(x) for x in stage_top[i, :compare_k])

                pulled = []
                for idx in stage_top[i, :change_pool]:
                    idx = int(idx)
                    rb = int(base_ranks[i, idx])
                    rs = int(stage_ranks[i, idx])
                    if idx not in base_compare or rb - rs >= compare_k:
                        pulled.append(entry(idx, rb, rs, base_sim[i, idx], stage_sim[i, idx]))
                    if len(pulled) >= change_k:
                        break

                pushed = []
                for idx in base_top[i, :change_pool]:
                    idx = int(idx)
                    rb = int(base_ranks[i, idx])
                    rs = int(stage_ranks[i, idx])
                    if idx not in stage_compare or rs - rb >= compare_k:
                        pushed.append(entry(idx, rb, rs, base_sim[i, idx], stage_sim[i, idx]))
                    if len(pushed) >= change_k:
                        break

                stable = []
                for idx in base_top[i, :change_pool]:
                    idx = int(idx)
                    if idx in stage_compare:
                        stable.append(entry(idx, int(base_ranks[i, idx]), int(stage_ranks[i, idx]), base_sim[i, idx], stage_sim[i, idx]))
                    if len(stable) >= change_k:
                        break

                pulled_all.append(pulled)
                pushed_all.append(pushed)
                stable_all.append(stable)
            changes[family][stage] = {"pulled": pulled_all, "pushed": pushed_all, "stable": stable_all}
            print(f"built neighbor changes for {family} {stage}")

    return coords, neighbors, changes


def read_model_meta(emb_dir: Path, model_keys: List[str]) -> Dict[str, Any]:
    out = {}
    for key in model_keys:
        meta_path = emb_dir / f"{key}.meta.json"
        out[key] = read_json(meta_path) if meta_path.exists() else IMAGE_MODELS[key]
    return out


def minified_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def concept_categories(concepts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen = set()
    for concept in concepts:
        cid = str(concept.get("category") or "uncategorized")
        if cid in seen:
            continue
        seen.add(cid)
        out.append({"id": cid, "label": str(concept.get("category_label") or cid)})
    return out or [{"id": "uncategorized", "label": "Uncategorized"}]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the model-centric ReasonCLIP explorer v3.")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--image-embedding-dir", required=True)
    parser.add_argument("--retrievals", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", default="all")
    parser.add_argument("--neighbor-k", type=int, default=18)
    parser.add_argument("--change-pool", type=int, default=80)
    parser.add_argument("--change-k", type=int, default=8)
    parser.add_argument("--compare-k", type=int, default=12)
    parser.add_argument("--thumb-size", type=int, default=224)
    parser.add_argument("--thumb-quality", type=int, default=82)
    parser.add_argument("--no-thumbs", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.annotations)
    ids = [str(r["image_id"]) for r in rows]
    emb_dir = Path(args.image_embedding_dir)
    saved_ids = read_json(emb_dir / "ids.json")
    if list(saved_ids) != ids:
        raise ValueError("annotation ids do not match image embedding ids")

    model_keys = [key for key in IMAGE_MODELS if (emb_dir / f"{key}.npy").exists()]
    if args.models != "all":
        requested = [m.strip() for m in args.models.split(",") if m.strip()]
        model_keys = [key for key in requested if key in model_keys]
    out_dir = Path(args.output_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)

    records = build_records(rows, out_dir, args.thumb_size, args.thumb_quality, make_thumbs=not args.no_thumbs)
    coords, neighbors, neighbor_changes = compute_neighbors_and_changes(emb_dir, model_keys, args.neighbor_k, args.change_pool, args.change_k, args.compare_k)
    retrievals = read_json(args.retrievals)
    categories = concept_categories(retrievals.get("concepts", []))
    data = {
        "schema_version": "v3",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "records": records,
        "models": read_model_meta(emb_dir, model_keys),
        "families": family_map(model_keys),
        "concepts": retrievals.get("concepts", []),
        "categories": categories,
        "retrievals": retrievals,
        "coords": coords,
        "neighbors": neighbors,
        "neighbor_changes": neighbor_changes,
    }
    data_dir = out_dir / "data"
    core = {k: data[k] for k in ["schema_version", "generated_at", "models", "families", "concepts", "categories"]}
    geometry = {k: data[k] for k in ["retrievals", "coords", "neighbors"]}
    changes = {"neighbor_changes": data["neighbor_changes"]}
    (data_dir / "explorer_core.js").write_text("window.EXPLORER_DATA=" + minified_json(core) + ";\n", encoding="utf-8")
    (data_dir / "explorer_records.js").write_text("window.EXPLORER_DATA.records=" + minified_json(data["records"]) + ";\n", encoding="utf-8")
    (data_dir / "explorer_geometry.js").write_text("Object.assign(window.EXPLORER_DATA," + minified_json(geometry) + ");\n", encoding="utf-8")
    (data_dir / "explorer_neighbor_changes.js").write_text("Object.assign(window.EXPLORER_DATA," + minified_json(changes) + ");\n", encoding="utf-8")
    (out_dir / "index.html").write_text(HTML, encoding="utf-8")
    print(f"wrote explorer v3 to {out_dir}")


if __name__ == "__main__":
    main()
