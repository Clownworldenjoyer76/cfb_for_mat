async function loadJSON(path){
  try{
    const res = await fetch(path + "?" + Date.now());
    return res.ok ? res.json() : [];
  }catch(_){ return []; }
}

function fmtDate(iso){ try{ return new Date(iso).toLocaleString(); }catch(_){ return iso || ""; } }

// ---------- Scores ----------
function renderScores(rows){
  const tbody = document.querySelector('#scoresTable tbody');
  tbody.innerHTML = '';
  rows.forEach(r=>{
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${fmtDate(r.kickoff_utc)}</td>
      <td>${r.away} @ ${r.home}</td>
      <td>${r.away_pred}</td>
      <td>${r.home_pred}</td>
      <td>${r.total}</td>
      <td>${r.home_spread}</td>
    `;
    tbody.appendChild(tr);
  });
}

// ---------- Picks (existing) ----------
function renderPicks(picks){
  const tbody = document.querySelector('#picksTable tbody');
  tbody.innerHTML = '';
  picks.forEach(p=>{
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${fmtDate(p.kickoff_utc)}</td>
      <td>${p.away} @ ${p.home}</td>
      <td>${p.market} ${p.selection}</td>
      <td>${p.line}</td>
      <td>${p.odds}</td>
      <td>${(p.model_prob*100).toFixed(1)}%</td>
    `;
    tbody.appendChild(tr);
  });
}

(async()=>{
  let scores = await loadJSON('scores.json');
  let picks  = await loadJSON('picks.json');

  // initial render
  renderScores(scores);
  renderPicks(picks);

  // refresh
  document.getElementById('refresh').addEventListener('click', async ()=>{
    scores = await loadJSON('scores.json');
    picks  = await loadJSON('picks.json');
    applySearchAndRender();
  });

  // search
  const searchInput = document.getElementById('search');
  searchInput.addEventListener('input', applySearchAndRender);

  function applySearchAndRender(){
    const q = searchInput.value.toLowerCase();
    const fs = scores.filter(r => (r.home + r.away).toLowerCase().includes(q));
    const fp = picks.filter(p => (p.home + p.away + p.selection).toLowerCase().includes(q));
    renderScores(fs);
    renderPicks(fp);
  }
})();
