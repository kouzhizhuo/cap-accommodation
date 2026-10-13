// Applied after fig1_teaser.figma.full.js (same frame 544:580): moves the lock chip / balance glyph below the "Ours" pill, bot icon 24 px, curves +2.
// To rebuild from scratch: run fig1_teaser.figma.full.js (helpers + body), then this patch. Excerpts are in const X at the top of the body.
const FR = await figma.getNodeByIdAsync('544:580');
const mv = (n, dy) => { n.y += dy; };
for (const n of FR.children) { if (n.x < 266 && n.type !== 'INSTANCE') continue;
  if (n.name === 'chip-can-but-wont' || (n.name === 'lock' && n.y > 300 && n.y < 340) || (n.type === 'TEXT' && n.characters === 'can, but won’t')) mv(n, 12);
  else if (n.name === 'glyph-reward-gap' || (n.type === 'TEXT' && n.characters === 'reward gap')) mv(n, 14);
  else if (n.name.startsWith('ours-curves')) mv(n, 2);
  else if (n.type === 'TEXT' && (n.characters === 'style' || n.characters === 'dialect')) mv(n, n.characters === 'style' ? 4 : 2);
  else if (n.type === 'TEXT' && n.characters === '0' && n.x > 490) mv(n, 2); }
const bot = FR.children.find(n => n.type === 'INSTANCE'); bot.rescale(24 / Math.max(bot.width, bot.height)); bot.x = 10; bot.y = 100;
