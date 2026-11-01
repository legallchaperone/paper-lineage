"""Interactive single-file HTML viewer.

Layout is computed here, not by a generic graph-layout engine: each line of work is a horizontal lane
(oldest on the left) flowing into the target on the right, so lines never tangle. Only the lines' main
chains (+ alternatives) are drawn by default; every other paper is reachable through the side panel.
No external scripts: works offline.
"""

from __future__ import annotations

import html
import json

from ..graph import LineageGraph

ROLE_SHORT = {"builds_on": "built on", "technique_source": "idea", "component": "uses",
              "alternative": "alternative", "problem_motivation": "motivates"}


def graph_payload(g: LineageGraph) -> dict:
    nodes = {}
    for k, n in g.nodes.items():
        m = n.meta
        nodes[k] = {
            "name": m.short_name, "title": m.title, "year": m.year, "url": m.url, "kind": n.kind,
            "cites": m.citation_count, "rel": None if n.relevance is None else round(n.relevance, 3),
            "note": n.note,
        }
    edges = [{"p": e.parent, "c": e.child, "role": e.role, "d": e.directness, "inh": e.inherited,
              "ev": e.evidence, "sec": e.section_kind} for e in g.edges
             if e.parent in g.nodes and e.child in g.nodes]
    lines = [{"label": line.label, "role": line.role, "head": line.head, "path": line.best_path,
              "members": line.members} for line in g.lines]
    return {"target": g.target, "spine": g.spine, "spineReason": g.spine_reason, "lines": lines,
            "nodes": nodes, "edges": edges, "stats": g.stats, "roles": ROLE_SHORT}


CSS = r"""
:root{--bg:#f7f6f3;--panel:#ffffff;--fg:#1c1917;--muted:#6b6560;--line:#e2ded8;--node:#ffffff;
 --nodeborder:#cfc9c1;--target:#fff1c2;--targetborder:#c2410c;--alt:#f3efff;--altborder:#8b74d6;
 --accent:#c2410c;--hl:#c2410c;--shadow:0 1px 2px rgba(0,0,0,.06),0 2px 8px rgba(0,0,0,.05);
 --l0:#2563eb;--l1:#0d9488;--l2:#c026d3;--l3:#d97706;--l4:#4f46e5;--l5:#dc2626}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#171513;--panel:#211e1b;--fg:#f2efea;
 --muted:#a39d96;--line:#3a3530;--node:#26221f;--nodeborder:#4a443e;--target:#3d2a12;--targetborder:#fb923c;
 --alt:#2a2440;--altborder:#a08ce6;--accent:#fb923c;--hl:#fb923c;--shadow:none;
 --l0:#60a5fa;--l1:#2dd4bf;--l2:#e879f9;--l3:#fbbf24;--l4:#818cf8;--l5:#f87171}}
:root[data-theme=dark]{--bg:#171513;--panel:#211e1b;--fg:#f2efea;--muted:#a39d96;--line:#3a3530;--node:#26221f;
 --nodeborder:#4a443e;--target:#3d2a12;--targetborder:#fb923c;--alt:#2a2440;--altborder:#a08ce6;--accent:#fb923c;
 --hl:#fb923c;--shadow:none;--l0:#60a5fa;--l1:#2dd4bf;--l2:#e879f9;--l3:#fbbf24;--l4:#818cf8;--l5:#f87171}
*{box-sizing:border-box}html,body{margin:0;height:100%}
body{background:var(--bg);color:var(--fg);font:14px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
 display:flex;flex-direction:column;overflow:hidden}
header{padding:12px 16px 8px;border-bottom:1px solid var(--line);background:var(--panel)}
h1{font-size:16px;margin:0 0 2px;line-height:1.3}
.sub{color:var(--muted);font-size:12.5px}.sub a{color:var(--accent)}
.toolbar{display:flex;flex-wrap:wrap;gap:6px 14px;align-items:center;margin-top:8px;font-size:13px}
.toolbar button{font:inherit;border:1px solid var(--line);background:var(--bg);color:var(--fg);border-radius:6px;
 padding:3px 10px;cursor:pointer}.toolbar button:hover{border-color:var(--accent)}
.toolbar label{display:flex;gap:5px;align-items:center;cursor:pointer;color:var(--muted)}
.toolbar input[type=search]{font:inherit;border:1px solid var(--line);background:var(--bg);color:var(--fg);
 border-radius:6px;padding:3px 8px;width:180px}
.legend{display:flex;flex-wrap:wrap;gap:4px 12px;margin-top:6px;font-size:12.5px}
.legend span{display:inline-flex;align-items:center;gap:5px;cursor:pointer}
.legend i{width:12px;height:4px;border-radius:2px;display:inline-block}
main{flex:1;display:flex;min-height:0}
#stage{flex:1;position:relative;min-width:0;touch-action:none;cursor:grab}
#stage.dragging{cursor:grabbing}
svg{width:100%;height:100%;display:block;user-select:none}
.node rect{fill:var(--node);stroke:var(--nodeborder);stroke-width:1.2}
.node .bar{stroke:none}
.node text{fill:var(--fg);font-size:13.5px}
.node .nm{font-weight:600}
.node .yr{fill:var(--muted);font-size:12px}
.pill{fill:var(--panel);stroke-width:1.2}.pilltx{font-size:11px;fill:var(--fg)}
.node{cursor:pointer}
.node:hover rect.box{stroke:var(--hl)}
.node.target rect.box{fill:var(--target);stroke:var(--targetborder);stroke-width:2.2}
.node.alt rect.box{fill:var(--alt);stroke:var(--altborder);stroke-dasharray:4 3}
.node.side rect.box{stroke-dasharray:2 3}
.node.sel rect.box{stroke:var(--hl);stroke-width:2.6}
.node.dim{opacity:.25}
.edge{fill:none;stroke-width:2}
.edge.alt{stroke:var(--altborder);stroke-dasharray:5 4;stroke-width:1.4}
.edge.cross,.edge.side{stroke:var(--muted);stroke-width:1.1;stroke-dasharray:3 4;opacity:.7}
.edge.short{stroke-width:1.4;stroke-dasharray:6 4}
.edge.dim{opacity:.12}
.elab{font-size:12px;fill:var(--muted)}
.lanelab{font-size:14px;font-weight:650}
.ghost{font-size:11px;fill:var(--muted)}
aside{width:400px;max-width:45vw;border-left:1px solid var(--line);background:var(--panel);overflow:auto;padding:14px 16px}
aside h2{font-size:15px;margin:0 0 4px;line-height:1.35}
aside h3{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin:16px 0 6px}
aside a{color:var(--accent)}
.meta{color:var(--muted);font-size:12.5px}
.badge{display:inline-block;font-size:11px;border:1px solid var(--line);border-radius:99px;padding:0 7px;margin-right:4px}
.rel{border-left:3px solid var(--line);padding:2px 0 2px 10px;margin:8px 0}
.rel .who{font-weight:600;cursor:pointer;color:var(--accent)}
.rel .who:hover{text-decoration:underline}
blockquote{margin:4px 0 0;color:var(--muted);font-size:12.5px;font-style:italic}
.chips{display:flex;flex-wrap:wrap;gap:6px}.chip{border:1px solid var(--line);border-radius:99px;padding:1px 9px;
 font-size:12.5px;cursor:pointer;background:var(--bg)}.chip:hover{border-color:var(--accent)}
.hint{color:var(--muted);font-size:12.5px}
#back{font:inherit;font-size:12px;border:1px solid var(--line);background:var(--bg);color:var(--fg);border-radius:6px;
 padding:1px 8px;cursor:pointer;float:right}
@media (max-width:760px){main{flex-direction:column}aside{width:auto;max-width:none;height:42vh;border-left:0;
 border-top:1px solid var(--line)}.toolbar input[type=search]{width:130px}}
"""

JS = r"""
const D = JSON.parse(document.getElementById('data').textContent);
const N = D.nodes, E = D.edges, T = D.target;
const NS = 'http://www.w3.org/2000/svg';
const LIN = new Set(['builds_on','technique_source','component']);
const W = 200, H = 58, COL = 252, PADX = 40, PADY = 60, ALTW = 184, ALTH = 40, ALTGAP = 8, ALTIN = 16;
const state = {alts: true, side: false, cross: false, sel: null, hist: []};
const parentsOf = k => E.filter(e => e.c === k && LIN.has(e.role));
const childrenOf = k => E.filter(e => e.p === k && LIN.has(e.role));
const altsOf = k => E.filter(e => e.p === k && e.role === 'alternative');
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const laneColor = i => `var(--l${i % 6})`;
const svg = document.getElementById('svg'), world = document.getElementById('world');
const el = (tag, attrs, parent) => { const x = document.createElementNS(NS, tag);
  for (const k in attrs) x.setAttribute(k, attrs[k]); if (parent) parent.appendChild(x); return x; };

function wrap(name, max) {             // split a name into <= 2 lines of ~max chars
  const words = String(name).split(/\s+/); const lines = [''];
  for (const w of words) { const cur = lines[lines.length - 1];
    if ((cur + ' ' + w).trim().length <= max) lines[lines.length - 1] = (cur + ' ' + w).trim();
    else if (lines.length < 2) lines.push(w); else { lines[1] = (lines[1] + ' ' + w); break; } }
  return lines.map(l => l.length > max ? l.slice(0, max - 1) + '…' : l);
}

let pos = {};      // key -> list of {x,y,w,h}
function layout() {
  pos = {};
  const lanes = D.lines.map((l, i) => ({i, label: l.label, chain: l.path.filter(k => k !== T)}));
  const maxLen = Math.max(1, ...lanes.map(l => l.chain.length));
  const items = [];  // drawable nodes
  const edges = [];
  let y = PADY;
  const onChain = new Set(lanes.flatMap(l => l.chain)); onChain.add(T);
  const laneY = [];
  for (const lane of lanes) {
    const sideRows = state.side ? lane.chain.map(k => parentsOf(k).filter(e => !onChain.has(e.p))) : [];
    const hasSide = state.side && sideRows.some(r => r.length);
    const nAlt = state.alts ? Math.max(0, ...lane.chain.map(k => Math.min(2, altsOf(k).length))) : 0;
    const top = y + (hasSide ? 52 : 0);
    laneY.push({lane, y: top, top: y});
    lane.chain.forEach((k, j) => {
      const col = maxLen - lane.chain.length + j;
      const x = PADX + col * COL;
      items.push({k, x, y: top, w: W, h: H, cls: 'chain', lane: lane.i});
      if (j > 0) edges.push({a: lane.chain[j - 1], b: k, lane: lane.i, cls: 'chain', ax: x - COL, ay: top, bx: x, by: top});
      if (state.alts) altsOf(k).slice(0, 2).forEach((e, ai) => {
        const ax = x + ALTIN, ay = top + H + 16 + ai * (ALTH + ALTGAP);
        items.push({k: e.c, x: ax, y: ay, w: ALTW, h: ALTH, cls: 'alt', lane: lane.i, of: k});
        edges.push({a: k, b: e.c, cls: 'alt', elbow: true, ax: x + 8, ay: top + H, bx: ax, by: ay + ALTH / 2});
      });
      if (state.side) sideRows[j].slice(0, 3).forEach((e, si) => {
        const sx = x + si * 70, sy = top - 46;
        items.push({k: e.p, x: sx, y: sy, w: 64, h: 26, cls: 'side', lane: lane.i, label: true});
        edges.push({a: e.p, b: k, cls: 'side', ax: sx + 32, ay: sy + 26, bx: x + 32 + si * 40, by: top});
      });
    });
    y = top + H + (nAlt ? 16 + nAlt * (ALTH + ALTGAP) : 0) + 50;
  }
  const tx = PADX + maxLen * COL + 30;
  const ty = laneY.length ? (laneY[0].y + laneY[laneY.length - 1].y) / 2 : PADY;
  items.push({k: T, x: tx, y: ty - 8, w: W + 24, h: H + 16, cls: 'target'});
  for (const {lane} of laneY) {
    const last = lane.chain[lane.chain.length - 1];
    if (last) edges.push({a: last, b: T, lane: lane.i, cls: 'chain', join: true});
  }
  // direct links into the target from chain nodes that are not their lane's head ("shortcuts")
  if (state.alts) altsOf(T).slice(0, 4).forEach((e, ai) => {
    const ax = tx + ALTIN + 8, ay = ty + H + 8 + 16 + ai * (ALTH + ALTGAP);
    items.push({k: e.c, x: ax, y: ay, w: ALTW, h: ALTH, cls: 'alt', of: T});
    edges.push({a: T, b: e.c, cls: 'alt', elbow: true, ax: tx + 8 + 8, ay: ty + H + 8, bx: ax, by: ay + ALTH / 2});
  });
  // descendants to the right of the target
  const desc = Object.keys(N).filter(k => N[k].kind === 'descendant');
  desc.forEach((k, di) => {
    const dx = tx + W + 24 + 90, dy = ty - (desc.length - 1) * 38 + di * 76;
    items.push({k, x: dx, y: dy, w: ALTW, h: ALTH, cls: 'desc'});
    edges.push({a: T, b: k, cls: 'chain', lane: 5});
  });
  for (const it of items) (pos[it.k] = pos[it.k] || []).push(it);
  // cross-lane lineage links between drawn chain nodes (shared history between lines)
  if (state.cross) for (const e of E) {
    if (!LIN.has(e.role) || e.c === T) continue;
    const A = (pos[e.p] || []).filter(p => p.cls === 'chain'), B = (pos[e.c] || []).filter(p => p.cls === 'chain');
    if (!A.length || !B.length) continue;
    if (A.some(a => B.some(b => a.lane === b.lane))) continue;   // drawn already within a lane
    edges.push({a: e.p, b: e.c, cls: 'cross', A: A[0], B: B[0]});
  }
  return {items, edges, laneY, maxLen, tx, ty};
}

function draw() {
  world.innerHTML = '';
  const L = layout();
  const gE = el('g', {}, world), gN = el('g', {}, world);
  for (const {lane, y, top} of L.laneY) {
    const first = lane.chain[0]; if (!first) continue;
    const x0 = PADX + (L.maxLen - lane.chain.length) * COL;
    const t = el('text', {x: x0, y: top - 12, class: 'lanelab', fill: laneColor(lane.i)}, gN);
    t.textContent = `Line ${lane.i + 1} · ${lane.label}` + (D.spine && D.lines[lane.i].path.join() === D.spine.join() ? '  ★ main path' : '');
  }
  const box = k => (pos[k] || [])[0];
  for (const e of L.edges) {
    let d, mx, my;
    if (e.cls === 'chain' && !e.join && e.ax !== undefined) {
      const x1 = e.ax + W, y1 = e.ay + H / 2, x2 = e.bx, y2 = e.by + H / 2;
      d = `M${x1},${y1} L${x2 - 6},${y2}`; mx = (x1 + x2) / 2; my = y1 - 6;
    } else if (e.cls === 'chain') {
      const A = box(e.a), B = box(e.b); if (!A || !B) continue;
      const x1 = A.x + A.w, y1 = A.y + A.h / 2, x2 = B.x, y2 = B.y + B.h / 2;
      const c = Math.max(40, (x2 - x1) / 2);
      d = `M${x1},${y1} C${x1 + c},${y1} ${x2 - c},${y2} ${x2 - 6},${y2}`; mx = x1 + 16; my = y1 - 6;
    } else if (e.cls === 'short') {
      const A = (pos[e.a] || []).find(p => p.lane === e.lane) || box(e.a), B = box(T); if (!A || !B) continue;
      const x1 = A.x + A.w / 2, y1 = A.y, x2 = B.x, y2 = B.y + B.h / 2 - 10;
      d = `M${x1},${y1} C${x1},${y1 - 34} ${x2 - 120},${y1 - 34} ${x2 - 6},${y2}`;
    } else if (e.cls === 'cross') {
      const x1 = e.A.x + W, y1 = e.A.y + H / 2, x2 = e.B.x, y2 = e.B.y + H / 2;
      d = `M${x1},${y1} C${x1 + 70},${y1} ${x2 - 70},${y2} ${x2 - 4},${y2}`;
    } else if (e.elbow) {
      d = `M${e.ax},${e.ay} L${e.ax},${e.by - 6} Q${e.ax},${e.by} ${e.ax + 6},${e.by} L${e.bx},${e.by}`;
    } else {
      d = `M${e.ax},${e.ay} L${e.bx},${e.by}`;
    }
    const color = e.cls === 'chain' || e.cls === 'short' ? laneColor(e.lane ?? 0) : null;
    const p = el('path', {d, class: `edge ${e.cls}`, 'data-a': e.a, 'data-b': e.b,
      'marker-end': e.cls === 'alt' ? '' : 'url(#arrow)'}, gE);
    if (color) p.style.stroke = color;
    if (e.cls === 'chain' && mx !== undefined) {
      const ed = E.find(x => x.p === e.a && x.c === e.b);
      if (ed) { const t = el('text', {x: mx, y: my, class: 'elab', 'text-anchor': e.join ? 'start' : 'middle'}, gE);
        t.textContent = D.roles[ed.role] || ed.role; }
    }
  }
  for (const it of L.items) {
    const n = N[it.k]; if (!n) continue;
    const cls = {chain: '', alt: 'alt', side: 'side', target: 'target', desc: 'alt'}[it.cls];
    const g = el('g', {class: `node ${cls}`, transform: `translate(${it.x},${it.y})`, 'data-k': it.k}, gN);
    el('rect', {class: 'box', width: it.w, height: it.h, rx: 8}, g);
    if (it.cls === 'chain') { const b = el('rect', {class: 'bar', width: 5, height: it.h - 2, x: 1, y: 1, rx: 3}, g);
      b.style.fill = laneColor(it.lane); }
    if (it.cls === 'side') {
      const t = el('text', {x: 8, y: 17, class: 'yr'}, g); t.textContent = wrap(n.name, 8)[0];
    } else {
      const max = it.cls === 'alt' || it.cls === 'desc' ? 24 : 27;
      const ls = wrap(n.name, max); const small = it.h < 50;
      ls.slice(0, small ? 1 : 2).forEach((s, i) => { const t = el('text', {x: 12, y: (small ? 18 : 20) + i * 16, class: 'nm'}, g); t.textContent = s; });
      const tag = {origin: ' · origin', frontier: ' · not traced further', foundation: ' · foundation',
                   alternative: ' · alternative', descendant: ' · builds on target', target: ' · this paper'}[n.kind] || '';
      const t = el('text', {x: 12, y: it.h - 9, class: 'yr'}, g); t.textContent = (n.year || '?') + tag;
    }
    if (it.cls === 'chain' && !(D.lines[it.lane] && D.lines[it.lane].head === it.k)) {
      const de = E.find(e => e.p === it.k && e.c === T && LIN.has(e.role));
      if (de) { const bw = 92; el('rect', {x: it.w - bw - 6, y: -9, width: bw, height: 18, rx: 9, class: 'pill'}, g).style.stroke = laneColor(it.lane);
        const bt = el('text', {x: it.w - bw / 2 - 6, y: 4, 'text-anchor': 'middle', class: 'pilltx'}, g);
        bt.textContent = '↘ also direct'; const t2 = el('title', {}, bt);
        t2.textContent = `${N[T].name} also uses this directly (${D.roles[de.role]}): ${de.inh || ''}`; }
    }
    const tt = el('title', {}, g); tt.textContent = n.title;
    g.addEventListener('click', ev => { ev.stopPropagation(); select(it.k, true); });
    g.addEventListener('mouseenter', () => highlight(it.k)); g.addEventListener('mouseleave', () => highlight(null));
  }
  if (state.sel) markSel();
  return L;
}

// ---- highlight: the hovered paper's route to the target + its direct parents
function routeToTarget(k) {
  const seen = new Set([k]); const stack = [k];
  while (stack.length) { const x = stack.pop(); for (const e of childrenOf(x)) if (!seen.has(e.c)) { seen.add(e.c); stack.push(e.c); } }
  return seen;
}
function highlight(k) {
  const nodes = world.querySelectorAll('.node'), edges = world.querySelectorAll('.edge');
  if (!k) { nodes.forEach(n => n.classList.remove('dim')); edges.forEach(e => e.classList.remove('dim')); return; }
  const keep = routeToTarget(k); parentsOf(k).forEach(e => keep.add(e.p)); altsOf(k).forEach(e => keep.add(e.c));
  nodes.forEach(n => n.classList.toggle('dim', !keep.has(n.dataset.k)));
  edges.forEach(e => e.classList.toggle('dim', !(keep.has(e.dataset.a) && keep.has(e.dataset.b))));
}

// ---- side panel
const panel = document.getElementById('panel');
function relRow(e, other, verb) {
  const n = N[other];
  return `<div class="rel" style="border-color:var(--line)"><span class="who" data-k="${esc(other)}">${esc(n ? n.name : other)}</span>
    ${n && n.year ? `<span class="meta">(${n.year})</span>` : ''} <span class="badge">${esc(verb)}</span>
    ${e.d !== undefined ? `<span class="badge">directness ${e.d}</span>` : ''}${e.sec ? `<span class="badge">${esc(e.sec)}</span>` : ''}
    ${e.inh ? `<div>${esc(e.inh)}</div>` : ''}${e.ev ? `<blockquote>“${esc(e.ev)}”</blockquote>` : ''}</div>`;
}
const KIND = {target: 'this paper', ancestor: 'ancestor', origin: 'origin of its line', frontier: 'not traced further (budget/depth)',
  foundation: 'generic foundation (not traced)', alternative: 'alternative approach', motivation: 'problem context',
  descendant: 'builds on the target', concurrent: 'concurrent work'};
function select(k, push) {
  if (push && state.sel && state.sel !== k) state.hist.push(state.sel);
  state.sel = k; const n = N[k]; if (!n) return;
  const ins = parentsOf(k), outs = childrenOf(k), alts = altsOf(k);
  const altOf = E.filter(e => e.c === k && e.role === 'alternative');
  const mot = E.filter(e => e.c === k && e.role === 'problem_motivation');
  const motOf = E.filter(e => e.p === k && e.role === 'problem_motivation');
  let h = (state.hist.length ? `<button id="back">← back</button>` : '') +
    `<h2>${n.url ? `<a href="${esc(n.url)}" target="_blank" rel="noopener">${esc(n.title)}</a>` : esc(n.title)}</h2>
     <div class="meta">${n.name !== n.title ? `<b>${esc(n.name)}</b> · ` : ''}${n.year || '?'}${n.cites != null ? ` · ${n.cites} citations` : ''}
     ${n.rel != null ? ` · similarity to target ${n.rel}` : ''}</div>
     <div style="margin-top:6px"><span class="badge">${esc(KIND[n.kind] || n.kind)}</span>${n.note ? `<span class="meta">${esc(n.note)}</span>` : ''}</div>`;
  const USED = {builds_on: 'builds on it', technique_source: 'takes its idea', component: 'uses it'};
  if (outs.length) h += `<h3>Why it is in the lineage</h3>` + outs.map(e => relRow(e, e.c, USED[e.role] || e.role)).join('');
  if (altOf.length) h += `<h3>Alternative to</h3>` + altOf.map(e => relRow(e, e.p, 'contrasted by')).join('');
  if (motOf.length) h += `<h3>Problem context for</h3>` + motOf.map(e => relRow(e, e.c, 'motivates')).join('');
  if (ins.length) h += `<h3>Its own parents (${ins.length})</h3>` + ins.map(e => relRow(e, e.p, D.roles[e.role])).join('');
  else if (k !== T && !['alternative','motivation','descendant','concurrent'].includes(n.kind))
    h += `<h3>Its own parents</h3><div class="hint">${n.kind === 'frontier' ? 'Not read (node budget or depth limit). Re-run with a larger --max-nodes / --max-depth.' : n.kind === 'foundation' ? 'Generic foundation: shown, not traced further.' : 'None kept: its references were off-topic, generic, or unavailable — this is where its line starts.'}</div>`;
  if (alts.length) h += `<h3>Done differently (${alts.length})</h3>` + alts.map(e => relRow(e, e.c, 'alternative')).join('');
  if (mot.length) h += `<h3>Problem context</h3>` + mot.map(e => relRow(e, e.p, 'motivates')).join('');
  panel.innerHTML = h;
  panel.scrollTop = 0;
  panel.querySelectorAll('.who').forEach(w => w.addEventListener('click', () => select(w.dataset.k, true)));
  const b = document.getElementById('back'); if (b) b.addEventListener('click', () => { const p = state.hist.pop(); state.sel = null; select(p, false); });
  markSel();
  if (pos[k]) centerOn(pos[k][0]);
}
function markSel() { world.querySelectorAll('.node').forEach(g => g.classList.toggle('sel', g.dataset.k === state.sel)); }
function overview() {
  const lines = D.lines.map((l, i) => `<div class="rel" style="border-color:${laneColor(i)}"><b style="color:${laneColor(i)}">Line ${i + 1} · ${esc(l.label)}</b><div>${
      l.path.map(k => `<span class="who" data-k="${esc(k)}">${esc(N[k].name)}</span>`).join(' → ')}</div></div>`).join('');
  const chips = kind => Object.keys(N).filter(k => N[k].kind === kind).map(k => `<span class="chip" data-k="${esc(k)}">${esc(N[k].name)}${N[k].year ? ' (' + N[k].year + ')' : ''}</span>`).join('');
  const sec = (title, kind, hint) => { const c = chips(kind); return c ? `<h3>${title}</h3>${hint ? `<div class="hint">${hint}</div>` : ''}<div class="chips">${c}</div>` : ''; };
  const s = D.stats || {};
  panel.innerHTML = `<h2>How to read this</h2>
    <div class="hint">Each colored lane is one line of work, oldest on the left, flowing into this paper on the right.
    Arrows are labelled with how the newer paper uses the older one. Dashed purple boxes are approaches the paper contrasts itself with.
    Click any paper for the sentences that justify each link. Drag to pan, scroll or pinch to zoom.</div>
    <h3>Lines</h3>${lines}
    ${sec('Problem context', 'motivation')}
    ${sec('Foundations (generic, not traced)', 'foundation')}
    ${sec('Also kept but not on a main chain', 'ancestor', 'Side parents and deeper ancestors. Toggle “side parents” to draw the direct ones.')}
    ${sec('Origins of side branches', 'origin')}
    ${sec('Not traced further (budget/depth)', 'frontier')}
    ${sec('Builds on this paper', 'descendant')}
    ${sec('Concurrent work', 'concurrent')}
    <h3>Run</h3><div class="hint">${s.expanded ?? '?'} papers read · ${s.classified_candidates ?? '?'} citations classified (${esc(s.classifier ?? '?')})${s.llm_calls ? ` · ${s.llm_calls} LLM calls` : ''}</div>`;
  panel.querySelectorAll('.who,.chip').forEach(w => w.addEventListener('click', () => select(w.dataset.k, true)));
}

// ---- pan / zoom
let view = {x: 0, y: 0, s: 1};
const apply = () => world.setAttribute('transform', `translate(${view.x},${view.y}) scale(${view.s})`);
function fit() {
  const bb = world.getBBox(), r = svg.getBoundingClientRect(); if (!bb.width) return;
  const full = Math.min((r.width - 30) / bb.width, (r.height - 30) / bb.height);
  const s = Math.min(1.15, Math.max(0.7, full));
  // if it can't all fit legibly, keep the target (right end) in view; drag left to see the origins
  const x = full >= 0.7 ? (r.width - bb.width * s) / 2 - bb.x * s : r.width - 20 - (bb.x + bb.width) * s;
  view = {s, x, y: Math.max(15 - bb.y * s, (r.height - bb.height * s) / 2 - bb.y * s)}; apply();
}
function centerOn(b) {
  const r = svg.getBoundingClientRect(); const s = Math.max(view.s, 0.85);
  view = {s, x: r.width / 2 - (b.x + b.w / 2) * s, y: r.height / 2 - (b.y + b.h / 2) * s}; apply();
}
function zoomAt(f, cx, cy) {
  const s = Math.min(3, Math.max(0.15, view.s * f)); const k = s / view.s;
  view = {s, x: cx - (cx - view.x) * k, y: cy - (cy - view.y) * k}; apply();
}
const stage = document.getElementById('stage');
stage.addEventListener('wheel', ev => { ev.preventDefault(); const r = svg.getBoundingClientRect();
  zoomAt(Math.exp(-ev.deltaY * (ev.ctrlKey ? 0.01 : 0.0015)), ev.clientX - r.left, ev.clientY - r.top); }, {passive: false});
const pointers = new Map(); let last = null, pinch = null, moved = false;
stage.addEventListener('pointerdown', ev => { pointers.set(ev.pointerId, ev); moved = false;
  if (pointers.size === 1) { last = {x: ev.clientX, y: ev.clientY}; stage.classList.add('dragging'); }
  if (pointers.size === 2) { const [a, b] = [...pointers.values()]; pinch = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY); } });
stage.addEventListener('pointermove', ev => { if (!pointers.has(ev.pointerId)) return; pointers.set(ev.pointerId, ev);
  if (pointers.size === 2 && pinch) { const [a, b] = [...pointers.values()]; const d = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
    const r = svg.getBoundingClientRect(); zoomAt(d / pinch, (a.clientX + b.clientX) / 2 - r.left, (a.clientY + b.clientY) / 2 - r.top); pinch = d; return; }
  if (last) { const dx = ev.clientX - last.x, dy = ev.clientY - last.y; if (Math.abs(dx) + Math.abs(dy) > 2) moved = true;
    view.x += dx; view.y += dy; last = {x: ev.clientX, y: ev.clientY}; apply(); } });
const up = ev => { pointers.delete(ev.pointerId); if (pointers.size < 2) pinch = null; if (!pointers.size) { last = null; stage.classList.remove('dragging'); } };
stage.addEventListener('pointerup', up); stage.addEventListener('pointercancel', up);
svg.addEventListener('click', () => { if (!moved) { state.sel = null; state.hist = []; markSel(); overview(); } });

// ---- controls
document.getElementById('fit').onclick = fit;
document.getElementById('zin').onclick = () => { const r = svg.getBoundingClientRect(); zoomAt(1.25, r.width / 2, r.height / 2); };
document.getElementById('zout').onclick = () => { const r = svg.getBoundingClientRect(); zoomAt(0.8, r.width / 2, r.height / 2); };
for (const id of ['alts', 'side', 'cross']) { const c = document.getElementById(id); c.checked = state[id];
  c.onchange = () => { state[id] = c.checked; draw(); fit(); }; }
document.getElementById('q').addEventListener('change', ev => {
  const q = ev.target.value.trim().toLowerCase(); if (!q) return;
  const k = Object.keys(N).find(k => N[k].name.toLowerCase().includes(q) || N[k].title.toLowerCase().includes(q));
  if (k) select(k, true); });
document.getElementById('legend').innerHTML = D.lines.map((l, i) =>
  `<span data-k="${esc(l.head)}"><i style="background:${laneColor(i)}"></i>${esc(l.label)}</span>`).join('') +
  `<span><i style="background:var(--altborder)"></i>alternative</span>`;
document.querySelectorAll('#legend span[data-k]').forEach(s => s.onclick = () => select(s.dataset.k, true));
window.addEventListener('resize', fit);
draw(); fit(); overview();
"""


def render_html(g: LineageGraph) -> str:
    tgt = g.nodes[g.target]
    payload = json.dumps(graph_payload(g), ensure_ascii=False).replace("</", "<\\/")
    spine = (f"Main path: {g.spine_reason}" if g.spine else f"No single main path: {g.spine_reason}")
    url = tgt.meta.url or ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lineage: {html.escape(tgt.meta.short_name)}</title><style>{CSS}</style></head>
<body>
<header>
  <h1>Lineage of {html.escape(tgt.meta.title)}</h1>
  <div class="sub">{f'<a href="{html.escape(url)}" target="_blank" rel="noopener">{html.escape(url)}</a> · ' if url else ''}{html.escape(spine)}</div>
  <div class="toolbar">
    <button id="fit" title="Fit to screen">Fit</button><button id="zin" title="Zoom in">+</button><button id="zout" title="Zoom out">−</button>
    <label><input type="checkbox" id="alts"> alternatives</label>
    <label><input type="checkbox" id="side"> side parents</label>
    <label><input type="checkbox" id="cross"> links between lines</label>
    <input type="search" id="q" placeholder="Find a paper…" aria-label="Find a paper">
  </div>
  <div class="legend" id="legend"></div>
</header>
<main>
  <div id="stage"><svg id="svg" role="img" aria-label="Lineage graph">
    <defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker></defs>
    <g id="world"></g></svg></div>
  <aside id="panel"></aside>
</main>
<script type="application/json" id="data">{payload}</script>
<script>{JS}</script>
</body></html>
"""
