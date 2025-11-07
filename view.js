// Mode: Run once for all items
// Ожидаем, что каждый item после HTTP к Qdrant содержит:
// json.ok = true/false, json.error (строка) при неуспехе
// имя файла берём из binary.data.fileName (или подстрахуемся)
const results = items.map((it, i) => {
  const bin = it.binary?.data;
  const name = bin?.fileName ?? it.json.filename ?? `file_${i+1}.pdf`;
  const ok = !!it.json.ok;
  const err = it.json.error || null;
  return { name, ok, err };
});

const total = results.length;
const succeeded = results.filter(r => r.ok).length;
const failed = total - succeeded;

const list = results.map(r =>
  `<li>${r.ok ? '✅' : '❌'} ${r.name}${r.err ? ' — ' + r.err : ''}</li>`
).join('');

const summary = `
<h2>Готово</h2>
<p>В Qdrant добавлено: <b>${succeeded}</b> из <b>${total}</b> файлов.</p>
<ul>${list}</ul>
`;

return [{ json: { total, succeeded, failed, summary_html: summary } }];
