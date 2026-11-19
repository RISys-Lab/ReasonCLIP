(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const R = window.RESULTS;
  const B = window.BENCH || { items: [] };
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const fmt = (v) => (v >= 1000 ? v.toFixed(1) : v.toFixed(1));

  /* ---------------- Nav + TOC + reveal ---------------- */
  const nav = $("#navbar");
  const onScroll = () => nav.classList.toggle("is-scrolled", window.scrollY > 24);
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  const tocLinks = $$(".toc a");
  const sections = tocLinks.map((a) => $(a.getAttribute("href")));
  const tocObs = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        tocLinks.forEach((a) => a.classList.toggle("is-active", a.getAttribute("href") === "#" + e.target.id));
      });
    },
    { rootMargin: "-40% 0px -55% 0px" }
  );
  sections.forEach((s) => s && tocObs.observe(s));

  const onVisible = (el, fn, opts = { threshold: 0.2 }) => {
    if (!el) return;
    const o = new IntersectionObserver((es) => {
      es.forEach((e) => {
        if (e.isIntersecting) {
          fn(e.target);
          o.unobserve(e.target);
        }
      });
    }, opts);
    o.observe(el);
  };
  $$(".reveal").forEach((el) => onVisible(el, (t) => t.classList.add("is-in"), { threshold: 0.12 }));

  // count-up on stats
  $$(".stat .value[data-count]").forEach((el) => {
    const target = parseFloat(el.dataset.count);
    if (reduceMotion) return;
    el.textContent = "+0.0";
    onVisible(el, () => {
      const t0 = performance.now();
      const dur = 1100;
      const step = (now) => {
        const p = Math.min(1, (now - t0) / dur);
        const e = 1 - Math.pow(1 - p, 3);
        el.textContent = "+" + (target * e).toFixed(1);
        if (p < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    });
  });

  /* ---------------- Category labels ---------------- */
  const CAT = {
    S: "Spatial / Geometric",
    A: "Attribute / State",
    H: "Creature / Action",
    C: "Creature / Action",
    T: "Temporal / Phase",
    P: "Intuitive Physics",
    SUBJ_FORM: "Subject form",
    NOUN_SWAP: "Noun swap",
    REL_PHRASE: "Relation",
    ATTR_STATE: "Attribute",
    SENT_STRUCT: "Sentence structure",
  };
  const catShort = (t) => (t === "H" ? "C" : t);

  /* ---------------- Hero morph: description -> reasoning ---------------- */
  (function morph() {
    const item = B.items.find((i) => i.id === B.hero) || B.items[0];
    const box = $("#morph");
    if (!item) {
      box.style.display = "none";
      return;
    }
    const steps = [{ label: "Description · what is visible", short: "Describe", text: item.desc }];
    (item.heroCats || ["A", "T", "P"]).forEach((c) => {
      const set = (item.tiers.v3 || []).find((s) => s.tag === c);
      if (set) steps.push({ label: `Reasoning · ${CAT[c]}`, short: catShort(c), text: set.gt });
    });
    $("#morphImg").src = item.img;
    const stepsEl = $("#morphSteps");
    stepsEl.innerHTML = steps.map((s, i) => `<button data-i="${i}">${esc(s.short)}</button>`).join("");
    const label = $("#morphLabel");
    const cap = $("#morphCaption");
    let idx = 0;
    let timer = null;
    let typing = null;
    let visible = true;

    const show = (i, auto) => {
      idx = i;
      $$("button", stepsEl).forEach((b, j) => b.classList.toggle("is-active", j === i));
      label.textContent = steps[i].label;
      const text = steps[i].text;
      clearInterval(typing);
      if (reduceMotion) {
        cap.textContent = text;
      } else {
        let n = 0;
        cap.innerHTML = '<span class="caret"></span>';
        typing = setInterval(() => {
          n += 2;
          cap.innerHTML = esc(text.slice(0, n)) + '<span class="caret"></span>';
          if (n >= text.length) clearInterval(typing);
        }, 18);
      }
      clearTimeout(timer);
      if (auto !== false && visible && !reduceMotion) timer = setTimeout(() => show((idx + 1) % steps.length), Math.max(5200, text.length * 18 + 3200));
    };
    stepsEl.addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      show(+b.dataset.i, false);
    });
    // pause auto-cycling while the hero is off screen
    new IntersectionObserver((es) => {
      es.forEach((e) => {
        const was = visible;
        visible = e.isIntersecting;
        if (!visible) clearTimeout(timer);
        else if (!was) show((idx + 1) % steps.length);
      });
    }).observe(box);
    show(0);
  })();

  /* ---------------- Method pipeline: steps <-> loss terms ---------------- */
  const EQ = {
    s1: String.raw`\begin{aligned}\mathcal{L}^{(1)} = \;& \htmlClass{tm tm-1}{\htmlClass{tm tm-3}{\lambda(t)}\,\mathcal{L}_{\text{align}}(x, T_b)} \\ +\;& \htmlClass{tm tm-2}{\htmlClass{tm tm-3}{\big(1-\lambda(t)\big)}\,\mathcal{L}_{\text{align}}(x, T_{rl})} \\ +\;& \htmlClass{tm tm-4}{\beta\,\lVert \theta-\theta_0 \rVert_2^2}\end{aligned}`,
    s2: String.raw`\begin{aligned}\mathcal{L}^{(2)} = \;& \htmlClass{tm tm-1}{\mathcal{L}_{\text{align}}(x, T_{rp})} \\ +\;& \htmlClass{tm tm-4}{\gamma\,\big(\htmlClass{tm tm-2}{\mathcal{L}_{\text{img-cls}}(g_v(x),\, \mathbf{y})} + \htmlClass{tm tm-3}{\mathcal{L}_{\text{txt-cls}}(g_t(T_{rp}),\, c)}\big)}\end{aligned}`,
  };
  const renderEqs = () =>
    $$(".eq[data-eq]").forEach((el) => {
      const tex = EQ[el.dataset.eq];
      if (window.katex) katex.render(tex, el, { displayMode: true, throwOnError: false, trust: true, strict: false });
      else el.textContent = tex;
    });
  if (window.katex) renderEqs();
  else window.addEventListener("load", renderEqs);

  (function pipe() {
    const stages = $$(".stage");
    const steps = stages.flatMap((st) => $$(".steps li", st).map((li) => ({ st, t: li.dataset.t })));
    const highlight = (st, t) => {
      stages.forEach((s) => {
        const on = s === st;
        $$("[data-t]", s).forEach((el) => el.classList.toggle("hl", on && el.dataset.t === t));
        const eq = $(".eq", s);
        $$(".tm", eq).forEach((m) => m.classList.toggle("hl", on && m.classList.contains("tm-" + t)));
        eq.classList.toggle("has-hl", on && !!t);
      });
    };
    let auto = 0, timer = null, paused = false;
    const tick = () => {
      if (paused) return;
      const s = steps[auto % steps.length];
      highlight(s.st, s.t);
      auto++;
      timer = setTimeout(tick, 2600);
    };
    stages.forEach((st) => {
      st.addEventListener("mouseover", (e) => {
        const el = e.target.closest("[data-t]");
        paused = true;
        clearTimeout(timer);
        if (el) highlight(st, el.dataset.t);
      });
      st.addEventListener("mouseleave", () => {
        highlight(null, null);
        paused = false;
        clearTimeout(timer);
        timer = setTimeout(tick, 2000);
      });
    });
    if (!reduceMotion) onVisible($("#pipe"), () => setTimeout(tick, 600), { threshold: 0.3 });
  })();

  /* ---------------- Stage 1 schedule chart ---------------- */
  const SCHED = {
    clip: { start: 0.6, mid: 0.3, end: 0.5 },
    siglip: { start: 0.7, mid: 0.5, end: 0.6 },
  };
  const T1 = 0.2, T2 = 0.8;
  let schedKey = "clip";
  const lam = (p, s) => (p <= T1 ? s.start : p <= T2 ? s.start + (s.mid - s.start) * ((p - T1) / (T2 - T1)) : s.end);
  const svg = $("#sched");
  const W = 460, H = 180, M = { l: 32, r: 8, t: 10, b: 24 };
  const X = (p) => M.l + p * (W - M.l - M.r);
  const Y = (v) => M.t + (1 - v) * (H - M.t - M.b);
  const pathFor = (fn) => {
    // piecewise: hold, linear shift, then a jump at T2 and a flat tail
    const ps = [0, T1, T2, T2 + 1e-6, 1];
    return ps.map((p, i) => `${i ? "L" : "M"}${X(p).toFixed(1)},${Y(fn(p)).toFixed(1)}`).join("");
  };
  const drawSched = () => {
    const s = SCHED[schedKey];
    const grid = [0, 0.5, 1]
      .map((v) => `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${M.l - 8}" y="${Y(v) + 4}" text-anchor="end">${v.toFixed(2)}</text>`)
      .join("");
    const xt = [0, 0.2, 0.4, 0.6, 0.8, 1].map((p) => `<text x="${X(p)}" y="${H - 8}" text-anchor="middle">${Math.round(p * 100)}%</text>`).join("");
    svg.innerHTML = `
      <rect class="phase" x="${X(0)}" y="${M.t}" width="${X(T1) - X(0)}" height="${H - M.t - M.b}"/>
      <rect class="phase" x="${X(T2)}" y="${M.t}" width="${X(1) - X(T2)}" height="${H - M.t - M.b}"/>
      <g class="grid">${grid}</g>${xt}
      <text x="${(X(0) + X(T1)) / 2}" y="${M.t + 14}" text-anchor="middle">hold</text>
      <text x="${(X(T1) + X(T2)) / 2}" y="${M.t + 14}" text-anchor="middle">shift to reasoning</text>
      <text x="${(X(T2) + X(1)) / 2}" y="${M.t + 14}" text-anchor="middle">settle</text>
      <path class="l-tb" d="${pathFor((p) => lam(p, s))}"/>
      <path class="l-tr" d="${pathFor((p) => 1 - lam(p, s))}"/>
      <line class="cursor" id="schedCursor" y1="${M.t}" y2="${H - M.b}" x1="-10" x2="-10"/>
      <circle class="knob" id="k1" r="4" cx="-10" cy="-10"/>
      <circle class="knob" id="k2" r="4" cx="-10" cy="-10"/>`;
    setCursor(0.5);
  };
  const setCursor = (p) => {
    const s = SCHED[schedKey];
    const l = lam(p, s);
    const c = $("#schedCursor", svg);
    if (!c) return;
    c.setAttribute("x1", X(p));
    c.setAttribute("x2", X(p));
    $("#k1", svg).setAttribute("cx", X(p));
    $("#k1", svg).setAttribute("cy", Y(l));
    $("#k2", svg).setAttribute("cx", X(p));
    $("#k2", svg).setAttribute("cy", Y(1 - l));
    $("#schedReadout").innerHTML = `${Math.round(p * 100)}% of training · T<sub>b</sub> <b>${l.toFixed(2)}</b> · T<sub>rl</sub> (dashed) <b>${(1 - l).toFixed(2)}</b>`;
  };
  const pointer = (e) => {
    const r = svg.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * W;
    setCursor(Math.min(1, Math.max(0, (x - M.l) / (W - M.l - M.r))));
  };
  svg.addEventListener("pointermove", pointer);
  svg.addEventListener("pointerdown", pointer);
  $("#schedTabs").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    schedKey = b.dataset.bb;
    $$("#schedTabs button").forEach((x) => x.classList.toggle("is-active", x === b));
    drawSched();
  });
  drawSched();

  /* ---------------- Category bars ---------------- */
  const catBars = $("#catBars");
  const maxCat = Math.max(...R.categories.map((c) => c[2]));
  catBars.innerHTML = R.categories
    .map(
      ([k, name, n]) => `
      <div class="cat-row">
        <span class="k"><b>${k}</b>${name}</span>
        <span class="track"><span class="fill" data-w="${((n / maxCat) * 100).toFixed(1)}"></span></span>
        <span class="v">${(n / 1e6).toFixed(2)}M</span>
      </div>`
    )
    .join("");
  onVisible(catBars, () => $$(".fill", catBars).forEach((f) => (f.style.width = f.dataset.w + "%")));

  /* ---------------- RCLIP-Bench browser ---------------- */
  (function bench() {
    const box = $("#benchBox");
    if (!B.items.length) {
      box.style.display = "none";
      return;
    }
    const TIERS = [
      ["v1", "V1 · Grounding"],
      ["v2", "V2 · Evidence"],
      ["v3", "V3 · Reasoning"],
    ];
    const state = { item: 0, tier: "v3", cat: null, answered: false, order: [] };

    // word-level diff: mark tokens of b that are not part of the LCS with a
    const diffMark = (a, b) => {
      const A = a.split(/(\s+)/), Bt = b.split(/(\s+)/);
      const n = A.length, m = Bt.length;
      const dp = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
      for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) dp[i][j] = A[i] === Bt[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
      let i = 0, j = 0;
      const toks = []; // [text, changed]
      while (j < m) {
        if (i < n && A[i] === Bt[j]) { toks.push([Bt[j], false]); i++; j++; }
        else if (i < n && dp[i + 1][j] >= dp[i][j + 1]) i++;
        else { toks.push([Bt[j], true]); j++; }
      }
      // whitespace between two changed words joins them into one underline
      toks.forEach((t, k) => {
        if (/^\s+$/.test(t[0]) && toks[k - 1] && toks[k + 1] && toks[k - 1][1] && toks[k + 1][1]) t[1] = true;
        else if (/^\s+$/.test(t[0])) t[1] = false;
      });
      let out = "", run = "";
      toks.forEach(([t, c]) => {
        if (c) run += t;
        else { if (run) { out += `<ins>${esc(run)}</ins>`; run = ""; } out += esc(t); }
      });
      if (run) out += `<ins>${esc(run)}</ins>`;
      return out;
    };

    const shuffle = (k) => {
      const a = [...Array(k).keys()];
      for (let i = a.length - 1; i > 0; i--) {
        const r = Math.floor(Math.random() * (i + 1));
        [a[i], a[r]] = [a[r], a[i]];
      }
      return a;
    };

    const sets = () => B.items[state.item].tiers[state.tier] || [];
    const curSet = () => sets().find((s) => s.tag === state.cat) || sets()[0];

    const renderTabs = () => {
      $("#tierTabs").innerHTML = TIERS.map(([k, l]) => `<button data-t="${k}" class="${k === state.tier ? "is-active" : ""}">${l}</button>`).join("");
      const tags = sets().map((s) => s.tag);
      if (!tags.includes(state.cat)) state.cat = tags[0];
      $("#catTabs").innerHTML = tags.map((t) => `<button data-c="${t}" class="${t === state.cat ? "is-active" : ""}">${esc(CAT[t] || t)}</button>`).join("");
    };

    const renderCands = () => {
      const s = curSet();
      const all = [s.gt, ...s.neg];
      const el = $("#cands");
      el.innerHTML = state.order
        .map((oi, pos) => {
          const isGt = oi === 0;
          let cls = "cand";
          let html = esc(all[oi]);
          if (state.answered) {
            if (isGt) cls += " is-gt";
            else {
              cls += state.picked === oi ? " is-wrong" : " is-dim";
              html = diffMark(s.gt, all[oi]);
            }
          }
          return `<button class="${cls}" data-o="${oi}" ${state.answered ? "disabled" : ""}><span class="idx">${String.fromCharCode(65 + pos)}</span><span>${html}</span></button>`;
        })
        .join("");
    };

    const render = (reshuffle = true) => {
      const it = B.items[state.item];
      $("#benchImg").src = it.img;
      $("#benchCounter").textContent = `${state.item + 1} / ${B.items.length}`;
      renderTabs();
      if (reshuffle) {
        state.answered = false;
        state.picked = null;
        state.order = shuffle(1 + curSet().neg.length);
      }
      const tierName = TIERS.find((t) => t[0] === state.tier)[1].split(" · ")[1];
      $("#benchQ").innerHTML = `<b>${tierName}</b> · ${esc(CAT[state.cat] || state.cat)}. Which sentence matches the image?`;
      $("#benchStatus").textContent = state.answered
        ? state.picked === 0
          ? "Correct. The negatives differ in the underlined words."
          : state.picked == null
          ? "The negatives differ in the underlined words."
          : "Not quite. The correct sentence is highlighted."
        : "Pick the sentence that matches the image.";
      $("#benchReveal").textContent = state.answered ? "Shuffle again" : "Reveal answer";
      renderCands();
    };

    $("#tierTabs").addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      state.tier = b.dataset.t;
      render();
    });
    $("#catTabs").addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      state.cat = b.dataset.c;
      render();
    });
    $("#cands").addEventListener("click", (e) => {
      const b = e.target.closest(".cand");
      if (!b || state.answered) return;
      state.answered = true;
      state.picked = +b.dataset.o;
      render(false);
    });
    $("#benchReveal").addEventListener("click", () => {
      if (state.answered) render(true);
      else {
        state.answered = true;
        state.picked = null;
        render(false);
      }
    });
    const go = (d) => {
      state.item = (state.item + d + B.items.length) % B.items.length;
      render();
    };
    $("#benchPrev").addEventListener("click", () => go(-1));
    $("#benchNext").addEventListener("click", () => go(1));
    render();
  })();

  /* ---------------- Results explorer ---------------- */
  (function explorer() {
    const state = { bb: "l14", task: "retrieval" };
    $("#bbTabs").innerHTML = R.backbones.map((b) => `<button data-k="${b.key}">${b.label}</button>`).join("");
    $("#taskTabs").innerHTML = R.tasks.map((t) => `<button data-k="${t.key}">${t.label}</button>`).join("");
    const bars = $("#bars");

    const render = () => {
      $$("#bbTabs button").forEach((b) => b.classList.toggle("is-active", b.dataset.k === state.bb));
      $$("#taskTabs button").forEach((b) => b.classList.toggle("is-active", b.dataset.k === state.task));
      const task = R.tasks.find((t) => t.key === state.task);
      const bb = R.backbones.find((b) => b.key === state.bb);
      const rows = R.data[state.bb][state.task];
      const vmax = Math.max(...rows.flat());
      const scale = (v) => (v / (vmax * 1.04)) * 100;
      const names = [bb.base, "+ Stage 1", "+ Stage 2"];
      const cls = ["base", "s1", "s2"];

      // keep existing nodes when the shape is the same so widths animate
      const fresh = bars.children.length !== rows.length || bars.dataset.task !== state.task;
      if (fresh) {
        bars.dataset.task = state.task;
        bars.innerHTML = rows
          .map(
            (_, i) => `
          <div class="bar-group">
            <div class="metric"><span>${task.metrics[i]}</span><span class="delta"></span></div>
            <div class="bar-lines">
              ${cls.map((c) => `<div class="bar-line ${c}"><span class="nm"></span><span class="track"><span class="fill"></span></span><span class="num"></span></div>`).join("")}
            </div>
          </div>`
          )
          .join("");
      }
      $$(".bar-group", bars).forEach((g, i) => {
        const vals = rows[i];
        const best = Math.max(vals[1], vals[2]);
        const d = best - vals[0];
        const delta = $(".delta", g);
        delta.textContent = `best ${d >= 0 ? "+" : "−"}${Math.abs(d).toFixed(1)}`;
        delta.classList.toggle("neg", d < 0);
        $$(".bar-line", g).forEach((line, j) => {
          $(".nm", line).textContent = names[j];
          const num = $(".num", line);
          num.textContent = fmt(vals[j]);
          num.classList.toggle("best", vals[j] === Math.max(...vals));
          const fill = $(".fill", line);
          if (fresh && !reduceMotion) {
            fill.style.width = "0";
            requestAnimationFrame(() => requestAnimationFrame(() => (fill.style.width = scale(vals[j]) + "%")));
          } else fill.style.width = scale(vals[j]) + "%";
        });
      });
      $("#explorerNote").textContent = task.note;
    };
    $("#bbTabs").addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      state.bb = b.dataset.k;
      render();
    });
    $("#taskTabs").addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      state.task = b.dataset.k;
      render();
    });
    onVisible(bars, render, { threshold: 0.05 });
    // render static state immediately in case observer is slow (e.g. deep link)
    if (location.hash === "#results") render();
  })();

  /* ---------------- MLLM table ---------------- */
  (function mllm() {
    const tb = $("#mllmTable tbody");
    const maxD = Math.max(...R.mllm.map(([n, a, b]) => (n === "MME" ? 0 : b - a)));
    tb.innerHTML = R.mllm
      .map(([n, a, b]) => {
        const d = b - a;
        const up = d > 0.05;
        const w = n === "MME" || d <= 0 ? 0 : Math.round((d / maxD) * 64);
        return `<tr><td>${n}</td><td>${fmt(a)}</td><td class="${up ? "up" : ""}">${fmt(b)}</td><td><span class="d ${up ? "up" : ""}">${d >= 0 ? "+" : "−"}${Math.abs(d).toFixed(1)}</span>${w ? `<span class="mini-bar" style="width:${w}px"></span>` : ""}</td></tr>`;
      })
      .join("");
  })();

  /* ---------------- Model zoo ---------------- */
  (function zoo() {
    const hf = (id) => (id ? `<a href="https://huggingface.co/RISys-Lab/${id}">${id}</a>` : "");
    const list = (...ids) => ids.filter(Boolean).map(hf).join("<br/>");
    $("#zooTable tbody").innerHTML = R.zoo
      .map(
        (z) => `<tr>
          <td><span style="font-weight:500">${z.label}</span><br/><span class="muted" style="font-size:11px">${z.base}</span></td>
          <td>${hf(z.s1)}</td><td>${list(z.s2, z.extra)}</td><td>${list(z.rea, z.des)}</td>
        </tr>`
      )
      .join("");
  })();

  /* ---------------- Copy buttons ---------------- */
  $$(".copy-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const text = $("#" + btn.dataset.copy).innerText;
      try {
        await navigator.clipboard.writeText(text);
        btn.textContent = "Copied";
      } catch {
        btn.textContent = "Press Ctrl+C";
        const r = document.createRange();
        r.selectNodeContents($("#" + btn.dataset.copy));
        const sel = getSelection();
        sel.removeAllRanges();
        sel.addRange(r);
      }
      setTimeout(() => (btn.textContent = "Copy"), 1600);
    });
  });
})();
