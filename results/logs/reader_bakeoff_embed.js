// Ternlight vectors for the reader bake-off (results/logs/reader_bakeoff.py).
//
//   TERNLIGHT_DIR=<dir with node_modules/@ternlight/{base,mini}> BAKEOFF_DIR=<work dir> \
//       node results/logs/reader_bakeoff_embed.js
//
// Writes vec_ternlight-<tier>.npy (float32, one 384-dim row per text, in
// texts.json order) plus a .meta.json with the published parameter count.
const fs = require('fs');
const path = require('path');

const work = process.env.BAKEOFF_DIR;
const texts = JSON.parse(fs.readFileSync(path.join(work, 'texts.json'), 'utf8'));
// the model card's figures: 2 layers, d=384 / 256, 30,522-word table
const PARAMS = { base: 15.4e6, mini: 9.5e6 };

function saveNpy(file, rows, dim) {
  const header = `{'descr': '<f4', 'fortran_order': False, 'shape': (${rows.length}, ${dim}), }`;
  let h = header;
  while ((10 + h.length + 1) % 64 !== 0) h += ' ';
  h += '\n';
  const pre = Buffer.alloc(10);
  pre.write('\x93NUMPY', 0, 'latin1');
  pre[6] = 1; pre[7] = 0;
  pre.writeUInt16LE(h.length, 8);
  const body = Buffer.alloc(rows.length * dim * 4);
  rows.forEach((v, i) => { for (let j = 0; j < dim; j++) body.writeFloatLE(v[j], (i * dim + j) * 4); });
  fs.writeFileSync(file, Buffer.concat([pre, Buffer.from(h, 'latin1'), body]));
}

for (const tier of ['base', 'mini']) {
  const mod = require(path.join(process.env.TERNLIGHT_DIR, 'node_modules', '@ternlight', tier));
  const t0 = process.hrtime.bigint();
  const rows = texts.map((t) => mod.embed(t));
  const ms = Number(process.hrtime.bigint() - t0) / 1e6;
  saveNpy(path.join(work, `vec_ternlight-${tier}.npy`), rows, rows[0].length);
  fs.writeFileSync(path.join(work, `vec_ternlight-${tier}.meta.json`),
    JSON.stringify({ params: PARAMS[tier], ms_per_text: ms / texts.length }));
  console.log(`ternlight-${tier}: ${texts.length} texts, ${(ms / texts.length).toFixed(2)} ms/text`);
}
