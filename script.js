async function loadPicks(){
  const res = await fetch('picks.json?' + Date.now());
  return res.ok ? res.json() : [];
}
function fmtDate(iso){try{return new Date(iso).toLocaleString()}catch{return iso}}
function render(picks){
  const tbody=document.querySelector('#picksTable tbody');
  tbody.innerHTML='';
  picks.forEach(p=>{
    const tr=document.createElement('tr');
    tr.innerHTML=`<td>${fmtDate(p.kickoff_utc)}</td>
      <td>${p.away} @ ${p.home}</td>
      <td>${p.market} ${p.selection}</td>
      <td>${p.line}</td>
      <td>${p.odds}</td>
      <td>${(p.model_prob*100).toFixed(1)}%</td>`;
    tbody.appendChild(tr);
  });
}
(async()=>{
  let picks=await loadPicks();
  render(picks);
  document.getElementById('refresh').addEventListener('click',async()=>{
    picks=await loadPicks();render(picks);
  });
  document.getElementById('search').addEventListener('input',e=>{
    const q=e.target.value.toLowerCase();
    render(picks.filter(p=>(p.home+p.away+p.selection).toLowerCase().includes(q)));
  });
})();