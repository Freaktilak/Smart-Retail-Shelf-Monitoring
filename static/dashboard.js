/**
 * ShelfIQ Dashboard — dashboard.js
 * Professional enterprise retail analytics frontend
 * Compatible with Python 3.8 / Jetson Nano backend
 */

'use strict';

/* ── Constants ──────────────────────────────────────────────── */
const GRID_ROWS   = 4;
const GRID_COLS   = 6;
const TOTAL_ZONES = GRID_ROWS * GRID_COLS;
const HISTORY_LEN = 60;

/* ── Helpers ─────────────────────────────────────────────────── */
const $  = id  => document.getElementById(id);
const $$ = sel => document.querySelector(sel);

/* ── Level classification ──────────────────────────────────── */
function levelOf(pct) {
  if (pct >= 75) return 'ok';
  if (pct >= 40) return 'warning';
  return 'critical';
}

function labelOf(pct) {
  if (pct >= 75) return 'Fully Stocked';
  if (pct >= 40) return 'Low Stock';
  return 'Critical — Restock Needed';
}

/* ── State ───────────────────────────────────────────────────── */
const history   = new Array(HISTORY_LEN).fill(null);
let   trendChart  = null;
let   prevLevel   = '';
let   logEntries  = [];   // local copy for clear-all

/* ═══════════════════════════════════════════════════════════════
   CHART
═══════════════════════════════════════════════════════════════ */
function initChart() {
  const ctx  = $('trend-chart').getContext('2d');
  const grad = ctx.createLinearGradient(0, 0, 0, 90);
  grad.addColorStop(0, 'rgba(37,99,235,0.14)');
  grad.addColorStop(1, 'rgba(37,99,235,0)');

  trendChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels:   new Array(HISTORY_LEN).fill(''),
      datasets: [{
        data:            [...history],
        borderColor:     '#2563EB',
        backgroundColor: grad,
        borderWidth:     1.5,
        pointRadius:     0,
        fill:            true,
        tension:         0.4,
        spanGaps:        false,
      }],
    },
    options: {
      animation:           false,
      responsive:          true,
      maintainAspectRatio: false,
      scales: {
        x: { display: false },
        y: {
          min: 0,
          max: 100,
          grid:  { color: '#F3F4F6', drawBorder: false },
          border: { display: false },
          ticks: {
            color:         '#9CA3AF',
            font:          { family: "'DM Sans'", size: 10 },
            callback:      v => v + '%',
            maxTicksLimit: 4,
            padding:       6,
          },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#111827',
          titleColor:      'transparent',
          bodyColor:       '#FFFFFF',
          bodyFont:        { family: "'Sora'", size: 11, weight: '600' },
          padding:         8,
          cornerRadius:    6,
          callbacks: {
            label:  ctx => ` ${ctx.parsed.y.toFixed(1)}% occupied`,
            title:  ()  => '',
          },
        },
      },
    },
  });
}

function pushHistory(val) {
  history.push(val);
  if (history.length > HISTORY_LEN) history.shift();
  if (trendChart) {
    // Update chart color based on level
    const level = levelOf(val);
    const colors = { ok: '#16A34A', warning: '#D97706', critical: '#DC2626' };
    trendChart.data.datasets[0].borderColor = colors[level] || '#2563EB';
    trendChart.data.datasets[0].data = [...history];
    trendChart.update('none');
  }
}

/* ═══════════════════════════════════════════════════════════════
   GRID INITIALISATION
═══════════════════════════════════════════════════════════════ */
function initZoneGrid() {
  const container = $('zone-grid');
  if (!container) return;
  container.innerHTML = '';
  container.style.gridTemplateColumns = `repeat(${GRID_COLS}, 1fr)`;

  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      const cell      = document.createElement('div');
      cell.className  = 'zone-cell';
      cell.id         = `zc-${r}-${c}`;
      cell.title      = `Row ${r + 1}, Zone ${c + 1}`;
      container.appendChild(cell);
    }
  }
}

function initRowTable() {
  const tbody = $('row-tbody');
  if (!tbody) return;
  tbody.innerHTML = '';

  for (let r = 0; r < GRID_ROWS; r++) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="row-name">Row ${r + 1}</td>
      <td id="rt-stk-${r}" style="font-variant-numeric:tabular-nums;font-size:12px;color:#374151;">—</td>
      <td>
        <div class="row-bar-wrap">
          <div class="row-mini-track">
            <div class="row-mini-fill ok" id="rt-bar-${r}" style="width:100%"></div>
          </div>
          <span class="row-pct-txt" id="rt-pct-${r}">—</span>
        </div>
      </td>
      <td style="text-align:right">
        <span class="row-status-tag ok" id="rt-tag-${r}">—</span>
      </td>
    `;
    tbody.appendChild(tr);
  }
}

/* ═══════════════════════════════════════════════════════════════
   STATUS SUMMARY CARD
═══════════════════════════════════════════════════════════════ */
function updateStatusCard(d) {
  const card   = $('status-card');
  const level  = levelOf(d.stock_pct);
  const label  = labelOf(d.stock_pct);

  card.innerHTML = `
    <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:10px">
      <div>
        <div class="sh-status">${label}</div>
        <div class="sh-detail" style="margin-top:3px">${d.stocked_count} of ${d.total_zones} zones occupied · ${d.empty_count} requiring restock</div>
      </div>
      <div style="text-align:right;flex-shrink:0">
        <div class="sh-pct">${d.stock_pct.toFixed(1)}%</div>
        <div class="sh-pct-sub">occupancy</div>
      </div>
    </div>
    <div class="sh-bar-track">
      <div class="sh-bar-fill" style="width:${d.stock_pct}%"></div>
    </div>
  `;
  card.className = `status-hero level-${level}`;
}

/* ═══════════════════════════════════════════════════════════════
   ZONE GRID UPDATE
═══════════════════════════════════════════════════════════════ */
function updateZones(grid) {
  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      const cell = $(`zc-${r}-${c}`);
      if (!cell) continue;
      const occupied = grid[r] && grid[r][c];
      const wanted   = `zone-cell ${occupied ? 'stocked' : 'empty'}`;
      if (cell.className !== wanted) cell.className = wanted;
    }
  }
}

/* ═══════════════════════════════════════════════════════════════
   ROW TABLE UPDATE
═══════════════════════════════════════════════════════════════ */
function updateRows(grid) {
  for (let r = 0; r < GRID_ROWS; r++) {
    const row    = grid[r] || [];
    const filled = row.filter(Boolean).length;
    const pct    = Math.round((filled / GRID_COLS) * 100);
    const level  = levelOf(pct);
    const tag    = level === 'ok' ? 'OK' : level === 'warning' ? 'Low Stock' : 'Empty';

    const stkEl  = $(`rt-stk-${r}`);
    const barEl  = $(`rt-bar-${r}`);
    const pctEl  = $(`rt-pct-${r}`);
    const tagEl  = $(`rt-tag-${r}`);

    if (stkEl) stkEl.textContent = `${filled} / ${GRID_COLS}`;
    if (barEl) { barEl.style.width = pct + '%'; barEl.className = `row-mini-fill ${level}`; }
    if (pctEl) pctEl.textContent = pct + '%';
    if (tagEl) { tagEl.textContent = tag; tagEl.className = `row-status-tag ${level}`; }
  }
}

/* ═══════════════════════════════════════════════════════════════
   RESTOCK RECOMMENDATIONS
═══════════════════════════════════════════════════════════════ */
function updateRestock(grid) {
  const list    = $('restock-list');
  const countEl = $('restock-count');
  if (!list || !countEl) return;

  const emptyRows = [];
  for (let r = 0; r < GRID_ROWS; r++) {
    const row     = grid[r] || [];
    const empty   = row.reduce((acc, v, i) => { if (!v) acc.push(i + 1); return acc; }, []);
    if (empty.length > 0) {
      emptyRows.push({ row: r + 1, empty, count: empty.length });
    }
  }

  if (emptyRows.length === 0) {
    countEl.textContent = '0 items';
    countEl.className   = 'restock-count badge-neutral';
    list.innerHTML      = `<div class="empty-state">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
        <path d="M22 11.08V12a10 10 0 11-5.93-9.14" stroke="currentColor" stroke-width="1.8"/>
        <polyline points="22 4 12 14.01 9 11.01" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
      </svg>
      All zones are adequately stocked
    </div>`;
    return;
  }

  countEl.textContent = `${emptyRows.reduce((s, r) => s + r.count, 0)} zones`;
  countEl.className   = 'restock-count badge-warn';

  list.innerHTML = emptyRows.map(item => `
    <div class="restock-item">
      <div class="restock-icon">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
          <rect x="2" y="3" width="20" height="15" rx="2" stroke="currentColor" stroke-width="1.8"/>
          <path d="M8 22h8M12 18v4" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
        </svg>
      </div>
      <span class="restock-text">Row ${item.row} — Zones ${item.empty.join(', ')} empty</span>
      <span class="restock-badge">${item.count} ${item.count === 1 ? 'zone' : 'zones'}</span>
    </div>
  `).join('');
}

/* ═══════════════════════════════════════════════════════════════
   ALERT STRIP
═══════════════════════════════════════════════════════════════ */
function updateAlert(d) {
  const strip = $('alert-strip');
  const text  = $('alert-text');
  const body  = $('page-body');
  const level = levelOf(d.stock_pct);

  if (level === 'ok') {
    strip.className = 'alert-strip hidden';
    body.classList.remove('shifted');
    return;
  }

  const msg = level === 'critical'
    ? `Critical: ${d.empty_count} zones are empty — immediate restock required.`
    : `Low stock: ${d.empty_count} zones running low. Schedule a restock.`;

  text.textContent = msg;
  strip.className  = `alert-strip level-${level}`;
  body.classList.add('shifted');
}

/* ═══════════════════════════════════════════════════════════════
   EVENT LOG
═══════════════════════════════════════════════════════════════ */
function updateEventLog(alerts) {
  if (!alerts || alerts.length === 0) return;
  logEntries = alerts;

  const log = $('event-log');
  if (!log) return;

  log.innerHTML = alerts.map(e => `
    <div class="log-row level-${e.level}">
      <span class="log-dot"></span>
      <span class="log-ts">${escHtml(e.ts)}</span>
      <span class="log-msg">${escHtml(e.msg)}</span>
    </div>
  `).join('');
}

/* ═══════════════════════════════════════════════════════════════
   MAIN UPDATE
═══════════════════════════════════════════════════════════════ */
function update(d) {
  const level = levelOf(d.stock_pct);

  // ── Nav bar ──────────────────────────────────────────
  setText('nav-uptime', formatUptime(d.uptime_s));
  setText('nav-temp',   d.cpu_temp);
  setText('nav-fps',    d.fps.toFixed(1) + ' fps');

  // ── Camera footer ────────────────────────────────────
  setText('feed-fps',   d.fps.toFixed(1));
  setText('feed-items', d.item_count);

  // ── KPI cards ────────────────────────────────────────
  setText('kv-occ',   d.stock_pct.toFixed(1) + '%');
  setText('kv-items', d.item_count);
  setText('kv-empty', d.empty_count);
  setText('kv-fps',   d.fps.toFixed(1));

  // Occupancy KPI sub-text
  setText('ks-occ', `${d.stocked_count} of ${d.total_zones} zones occupied`);

  // Empty zones sub-text
  const emEl = $('ks-empty');
  if (emEl) {
    emEl.textContent = d.empty_count === 0
      ? 'No empty zones'
      : `${d.empty_count} ${d.empty_count === 1 ? 'zone requires' : 'zones require'} restock`;
  }

  // Occupancy bar on KPI card
  const kbOcc = $('kb-occ');
  if (kbOcc) {
    kbOcc.style.width    = d.stock_pct + '%';
    kbOcc.className      = `kpi-bar-fill kpi-bar-${level === 'ok' ? 'green' : level === 'warning' ? 'amber' : 'red'}`;
  }

  // KPI empty card icon color
  const kpiEmCard = $('kpi-empty-card');
  if (kpiEmCard) {
    const icon = kpiEmCard.querySelector('.kpi-icon-wrap');
    if (icon) {
      icon.className = `kpi-icon-wrap ${d.empty_count === 0 ? 'kpi-green' : d.empty_count <= 3 ? 'kpi-amber' : 'kpi-red'}`;
    }
  }

  // ── Status summary ───────────────────────────────────
  updateStatusCard(d);

  // ── Zone heatmap ─────────────────────────────────────
  if (d.grid) {
    updateZones(d.grid);
    updateRows(d.grid);
    updateRestock(d.grid);
    setText('zone-summary',
      `${d.stocked_count} stocked · ${d.empty_count} empty · ${d.total_zones} total zones`);
  }

  // ── Alert strip ──────────────────────────────────────
  updateAlert(d);

  // ── Trend chart ──────────────────────────────────────
  pushHistory(d.stock_pct);

  // ── Event log ────────────────────────────────────────
  if (d.alert_log) updateEventLog(d.alert_log);
}

/* ═══════════════════════════════════════════════════════════════
   SSE CONNECTION
═══════════════════════════════════════════════════════════════ */
function connectSSE() {
  const es = new EventSource('/stats_stream');

  es.onopen = () => {
    setConn(true);
    console.log('[ShelfIQ] Stream connected');
  };

  es.onmessage = e => {
    try { update(JSON.parse(e.data)); }
    catch (err) { console.warn('[ShelfIQ] Parse error:', err); }
  };

  es.onerror = () => {
    setConn(false);
    es.close();
    console.warn('[ShelfIQ] Stream lost — reconnecting in 3 s');
    setTimeout(connectSSE, 3000);
  };
}

function setConn(online) {
  const chip  = $('conn-chip');
  const label = $('conn-label');
  if (chip)  chip.className  = `conn-chip ${online ? 'online' : 'offline'}`;
  if (label) label.textContent = online ? 'Connected' : 'Reconnecting…';
}

/* ═══════════════════════════════════════════════════════════════
   SNAPSHOT
═══════════════════════════════════════════════════════════════ */
$('btn-snapshot')?.addEventListener('click', async () => {
  try {
    const res  = await fetch('/api/snapshot', { method: 'POST' });
    const data = await res.json();
    showToast(data.file ? `Snapshot saved: ${data.file}` : 'Snapshot failed');
  } catch {
    showToast('Error saving snapshot');
  }
});

/* ═══════════════════════════════════════════════════════════════
   CLEAR LOG
═══════════════════════════════════════════════════════════════ */
$('btn-clear-log')?.addEventListener('click', () => {
  logEntries = [];
  const log = $('event-log');
  if (log) log.innerHTML = '<div class="empty-state">Log cleared</div>';
});

/* ═══════════════════════════════════════════════════════════════
   UTILITIES
═══════════════════════════════════════════════════════════════ */
function setText(id, val) {
  const el = $(id);
  if (el && el.textContent !== String(val)) el.textContent = val;
}

function formatUptime(secs) {
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = secs % 60;
  return [h, m, s].map(v => String(v).padStart(2, '0')).join(':');
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function showToast(msg) {
  const host  = $('toast-host');
  const toast = document.createElement('div');
  toast.className   = 'toast';
  toast.textContent = msg;
  host.appendChild(toast);
  requestAnimationFrame(() => requestAnimationFrame(() => toast.classList.add('show')));
  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 250);
  }, 3000);
}

/* ═══════════════════════════════════════════════════════════════
   BOOT
═══════════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  initZoneGrid();
  initRowTable();
  initChart();

  // Initial snapshot so the page isn't blank on load
  fetch('/api/stats')
    .then(r => r.json())
    .then(d => update(d))
    .catch(() => {});

  connectSSE();
});
