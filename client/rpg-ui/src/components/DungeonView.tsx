// The dungeon, drawn on a canvas. Everything here is painted from arrays in
// code — no sprite sheets, no third-party or lifted game art. The sprites are
// 16x16 pixel grids defined below and blown up 3x with smoothing off; the
// terrain is procedural stonework keyed off a hash of the tile coordinate, so
// every wall block is a little different but the same every frame.
//
// The human sees the whole map; the lit pool outlines the 5x5 the model was
// actually shown this turn, so it is obvious when a bad move was a bad
// decision rather than a blind one. Everything outside that window is drawn
// but dimmed — visible to you, dark to the model.
import { useEffect, useRef } from 'react';
import { player, type RpgState } from '../lib/rpgApi';

const TILE = 48;
const SPRITE = 16;
const MOVE_MS = 170;
const FLASH_MS = 320;

// ---------------------------------------------------------------- sprites

type Palette = Record<string, string>;

interface Sprite {
  rows: string[];
  palette: Palette;
}

const HERO: Sprite = {
  palette: {
    p: '#c2472f', // plume
    H: '#7c8798', // steel shadow
    h: '#ccd4e0', // steel
    f: '#2b2320', // visor slit
    m: '#d8a53a', // gold trim
    t: '#3d9160', // tunic
    T: '#276445', // tunic shadow
    d: '#4a7fbe', // shield face
    D: '#2f5b8c', // shield rim
    w: '#eef3f8', // blade
    g: '#7a5432', // grip
    b: '#3a2b22', // boots
  },
  rows: [
    '.......pp.......',
    '......pppp......',
    '....HHHHHHHH....',
    '....HhhhhhhH....',
    '....HffffffH....',
    '....HhhhhhhHw...',
    '...mmttttttmw...',
    '..dDtttttttTw...',
    '..dDtttmtttTw...',
    '..dDTTTmTTTTw...',
    '...D.TTTTTT.g...',
    '.....TTTTTT.....',
    '.....TT..TT.....',
    '.....TT..TT.....',
    '....bbb..bbb....',
    '....bbb..bbb....',
  ],
};

const GOBLIN: Sprite = {
  palette: {
    G: '#74a544', // skin
    g: '#4d7229', // skin shadow
    e: '#f2d14a', // eye
    E: '#2a1a10', // pupil
    t: '#f0ead8', // tusk
    r: '#6b4a2f', // rags
    k: '#c3ccd8', // dagger
  },
  rows: [
    '................',
    '................',
    '..g..GGGGGG..g..',
    '..ggGGGGGGGGgg..',
    '...GGeEGGeEGG...',
    '...GGGGGGGGGG...',
    '....GtGGGGtG....',
    '....gGGGGGGg....',
    '.....rrrrrr..k..',
    '...GGrrrrrrGk...',
    '...GGrrrrrrGk...',
    '.....rrrrrr.....',
    '.....gg..gg.....',
    '.....gg..gg.....',
    '....ggg..ggg....',
    '................',
  ],
};

const KEY: Sprite = {
  palette: { K: '#e8bd4c', k: '#9d7420', w: '#fff0b8' },
  rows: [
    '................',
    '................',
    '................',
    '...kkkk.........',
    '..kKwwKk........',
    '..kK..Kkkkkkk...',
    '..kK..KKKKKKK...',
    '..kKwwKk...K.K..',
    '...kkkk....K.K..',
    '...........k.k..',
    '................',
    '................',
    '................',
    '................',
    '................',
    '................',
  ],
};

const POTION: Sprite = {
  palette: {
    c: '#8b6236', // cork
    G: '#cfe3ef', // glass
    l: '#e2568a', // liquid
    L: '#a32f59', // liquid shadow
    w: '#ffffff', // shine
  },
  rows: [
    '................',
    '................',
    '................',
    '.......cccc.....',
    '.......cccc.....',
    '.......GGGG.....',
    '......GGGGGG....',
    '.....GGllllGG...',
    '....GGllllllGG..',
    '....GwllllllLG..',
    '....GwllllllLG..',
    '....GGLLLLLLGG..',
    '.....GGLLLLGG...',
    '......GGGGGG....',
    '................',
    '................',
  ],
};

/** Sprites are rasterised once at 1:1 and then blitted scaled, so a frame
 *  costs a handful of drawImage calls instead of a few thousand fillRects. */
interface Baked {
  art: HTMLCanvasElement;
  flip: HTMLCanvasElement;
  glow: HTMLCanvasElement;
  glowFlip: HTMLCanvasElement;
}

function bake(sprite: Sprite, tint: string): Baked {
  const make = (flip: boolean, solid: string | null) => {
    const c = document.createElement('canvas');
    c.width = SPRITE;
    c.height = SPRITE;
    const g = c.getContext('2d')!;
    sprite.rows.forEach((row, y) => {
      for (let x = 0; x < row.length; x++) {
        const ch = row[x];
        if (ch === '.') continue;
        g.fillStyle = solid ?? sprite.palette[ch] ?? '#f0f';
        g.fillRect(flip ? SPRITE - 1 - x : x, y, 1, 1);
      }
    });
    return c;
  };
  return {
    art: make(false, null),
    flip: make(true, null),
    glow: make(false, tint),
    glowFlip: make(true, tint),
  };
}

// ---------------------------------------------------------------- terrain

/** Deterministic per-tile noise: the same wall is chipped the same way every
 *  frame and every reload. */
function hash(x: number, y: number, salt = 0): number {
  let h = (x * 374761393 + y * 668265263 + salt * 2246822519) | 0;
  h = (h ^ (h >>> 13)) * 1274126177;
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

// Walls are cool and dark, floors warm and light: the two must be tellable
// apart at a glance, dimmed, at the far end of the map.
const STONE = {
  deep: '#0a0d11',
  block: '#2c3641',
  blockLit: '#3a4854',
  blockDark: '#1e262e',
  mortar: '#0b0e12',
  moss: '#3f6b45',
  floor: '#8a7c66',
  floorAlt: '#7d7059',
  grout: '#3d3529',
  grit: '#a2937a',
  crack: '#544a39',
};

function paintWall(ctx: CanvasRenderingContext2D, x: number, y: number, floorBelow: boolean) {
  const px = x * TILE;
  const py = y * TILE;
  ctx.fillStyle = STONE.mortar;
  ctx.fillRect(px, py, TILE, TILE);

  // two courses of brick, offset on alternating rows
  const courses = 2;
  const bh = TILE / courses;
  for (let c = 0; c < courses; c++) {
    const offset = (y + c) % 2 ? TILE / 4 : -TILE / 4;
    for (let b = -1; b < 3; b++) {
      const bx = px + offset + b * (TILE / 2);
      const by = py + c * bh;
      const w = TILE / 2 - 2;
      const h = bh - 2;
      const n = hash(x * 3 + b, y * 3 + c);
      ctx.fillStyle = n > 0.72 ? STONE.blockLit : n < 0.22 ? STONE.blockDark : STONE.block;
      ctx.fillRect(bx + 1, by + 1, w, h);
      // bevel: lit top edge, dark bottom edge
      ctx.fillStyle = 'rgba(255,255,255,0.09)';
      ctx.fillRect(bx + 1, by + 1, w, 2);
      ctx.fillStyle = 'rgba(0,0,0,0.28)';
      ctx.fillRect(bx + 1, by + h - 1, w, 2);
      if (n > 0.88) {
        ctx.fillStyle = STONE.blockDark;
        ctx.fillRect(bx + 4 + n * 8, by + 5, 5, 2);
      }
      if (hash(x + b, y + c, 7) > 0.9) {
        ctx.fillStyle = STONE.moss;
        ctx.globalAlpha = 0.5;
        ctx.fillRect(bx + 2, by + h - 5, w * 0.6, 4);
        ctx.globalAlpha = 1;
      }
    }
  }
  // a cap of lighter stone along the top of the block, and a hard dark seam
  // all the way round, so a wall never reads as a floor tile
  ctx.fillStyle = 'rgba(150,180,205,0.10)';
  ctx.fillRect(px, py, TILE, 4);
  ctx.strokeStyle = 'rgba(0,0,0,0.7)';
  ctx.lineWidth = 2;
  ctx.strokeRect(px + 1, py + 1, TILE - 2, TILE - 2);
  if (floorBelow) {
    ctx.fillStyle = 'rgba(0,0,0,0.45)';
    ctx.fillRect(px, py + TILE - 4, TILE, 4);
  }
}

function paintFloor(ctx: CanvasRenderingContext2D, x: number, y: number) {
  const px = x * TILE;
  const py = y * TILE;
  ctx.fillStyle = STONE.grout;
  ctx.fillRect(px, py, TILE, TILE);
  // 2x2 flagstones per tile, each nudged in tone
  for (let sy = 0; sy < 2; sy++) {
    for (let sx = 0; sx < 2; sx++) {
      const n = hash(x * 2 + sx, y * 2 + sy, 3);
      ctx.fillStyle = n > 0.5 ? STONE.floor : STONE.floorAlt;
      ctx.fillRect(px + sx * (TILE / 2) + 1, py + sy * (TILE / 2) + 1, TILE / 2 - 2, TILE / 2 - 2);
      ctx.fillStyle = 'rgba(255,255,255,0.06)';
      ctx.fillRect(px + sx * (TILE / 2) + 1, py + sy * (TILE / 2) + 1, TILE / 2 - 2, 1);
      if (n > 0.86) {
        ctx.fillStyle = STONE.crack;
        ctx.fillRect(px + sx * (TILE / 2) + 5, py + sy * (TILE / 2) + 9, 9, 1);
      }
      if (n < 0.1) {
        ctx.fillStyle = STONE.grit;
        ctx.fillRect(px + sx * (TILE / 2) + 14, py + sy * (TILE / 2) + 6, 2, 2);
      }
    }
  }
}

function paintStairs(ctx: CanvasRenderingContext2D, x: number, y: number) {
  const px = x * TILE;
  const py = y * TILE;
  // a stairwell in one-point perspective: each step is narrower and darker
  // than the one in front of it, so the tile reads as going *down*
  ctx.fillStyle = '#05070a';
  ctx.fillRect(px + 2, py + 2, TILE - 4, TILE - 4);
  const steps = 5;
  for (let i = 0; i < steps; i++) {
    const t = i / steps;
    const w = (TILE - 8) * (1 - t * 0.55);
    const sx = px + TILE / 2 - w / 2;
    const sy = py + TILE - 6 - i * 7.5;
    ctx.fillStyle = `rgba(188,208,228,${0.9 - t * 0.62})`;
    ctx.fillRect(sx, sy, w, 5);
    ctx.fillStyle = `rgba(0,0,0,${0.45 + t * 0.2})`;
    ctx.fillRect(sx, sy + 5, w, 2.5);
  }
  ctx.strokeStyle = 'rgba(0,0,0,0.75)';
  ctx.lineWidth = 2;
  ctx.strokeRect(px + 3, py + 3, TILE - 6, TILE - 6);
}

/** Torches go on wall tiles that face open floor — sparse, deterministic. */
function torchSpots(state: RpgState): { x: number; y: number }[] {
  const { width, height, rows } = state.map;
  const out: { x: number; y: number }[] = [];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if ((rows[y]?.[x] ?? '#') !== '#') continue;
      if ((rows[y + 1]?.[x] ?? '#') === '#') continue;
      if (hash(x, y, 11) > 0.62) out.push({ x, y });
    }
  }
  return out;
}

function buildTerrain(state: RpgState): HTMLCanvasElement {
  const { width, height, rows } = state.map;
  const c = document.createElement('canvas');
  c.width = width * TILE;
  c.height = height * TILE;
  const ctx = c.getContext('2d')!;
  ctx.imageSmoothingEnabled = false;
  ctx.fillStyle = STONE.deep;
  ctx.fillRect(0, 0, c.width, c.height);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const tile = rows[y]?.[x] ?? '#';
      if (tile === '#') continue;
      paintFloor(ctx, x, y);
      if (tile === 'E') paintStairs(ctx, x, y);
    }
  }
  // walls last so their shadows fall over the floor
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if ((rows[y]?.[x] ?? '#') !== '#') continue;
      paintWall(ctx, x, y, (rows[y + 1]?.[x] ?? '#') !== '#');
      if ((rows[y + 1]?.[x] ?? '#') !== '#') {
        const g = ctx.createLinearGradient(0, (y + 1) * TILE, 0, (y + 1) * TILE + 14);
        g.addColorStop(0, 'rgba(0,0,0,0.5)');
        g.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = g;
        ctx.fillRect(x * TILE, (y + 1) * TILE, TILE, 14);
      }
    }
  }
  // torch brackets are static; the flame is drawn per frame
  for (const t of torchSpots(state)) {
    const px = t.x * TILE + TILE / 2;
    const py = t.y * TILE + TILE * 0.62;
    ctx.fillStyle = '#241c14';
    ctx.fillRect(px - 3, py, 6, 12);
    ctx.fillStyle = '#3b2f22';
    ctx.fillRect(px - 5, py - 3, 10, 4);
  }
  return c;
}

// ---------------------------------------------------------------- component

interface Motion {
  x: number;
  y: number;
  fromX: number;
  fromY: number;
  t0: number;
  face: 1 | -1;
  hp: number;
  hitAt: number;
}

interface Props {
  state: RpgState;
}

export function DungeonView({ state }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const stateRef = useRef(state);
  const terrainRef = useRef<{ key: string; canvas: HTMLCanvasElement } | null>(null);
  const darkRef = useRef<HTMLCanvasElement | null>(null);
  const motionRef = useRef<Map<string, Motion>>(new Map());
  const bakedRef = useRef<Record<string, Baked> | null>(null);

  if (!bakedRef.current) {
    bakedRef.current = {
      hero: bake(HERO, '#ff5a4a'),
      goblin: bake(GOBLIN, '#ff5a4a'),
      key: bake(KEY, '#fff3c4'),
      potion: bake(POTION, '#ffd0e2'),
    };
  }

  // Fold the new state into the motion table: anything that moved gets a
  // tween from where it currently *looks* like it is, so back-to-back turns
  // don't snap.
  stateRef.current = state;
  useEffect(() => {
    const now = performance.now();
    const m = motionRef.current;
    const seen = new Set<string>();
    const track = (id: string, x: number, y: number, hp: number) => {
      seen.add(id);
      const prev = m.get(id);
      if (!prev) {
        m.set(id, { x, y, fromX: x, fromY: y, t0: now - MOVE_MS, face: 1, hp, hitAt: 0 });
        return;
      }
      if (prev.x !== x || prev.y !== y) {
        const p = Math.min(1, (now - prev.t0) / MOVE_MS);
        prev.fromX = prev.fromX + (prev.x - prev.fromX) * p;
        prev.fromY = prev.fromY + (prev.y - prev.fromY) * p;
        prev.t0 = now;
        if (x !== prev.x) prev.face = x > prev.x ? 1 : -1;
        prev.x = x;
        prev.y = y;
      }
      if (hp < prev.hp) prev.hitAt = now;
      prev.hp = hp;
    };
    const p = player(state);
    track(p.id, p.x, p.y, p.hp);
    for (const e of state.entities.enemy) track(e.id, e.x, e.y, e.hp);
    for (const [id] of m) if (!seen.has(id)) m.delete(id);
  }, [state]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const baked = bakedRef.current!;
    let raf = 0;

    const draw = (now: number) => {
      raf = requestAnimationFrame(draw);
      const st = stateRef.current;
      const { width, height } = st.map;
      const W = width * TILE;
      const H = height * TILE;

      const dpr = window.devicePixelRatio || 1;
      if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
        canvas.width = W * dpr;
        canvas.height = H * dpr;
        canvas.style.width = `${W}px`;
        canvas.style.height = `${H}px`;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.imageSmoothingEnabled = false;

      const key = `${st.scenario}:${width}x${height}:${st.map.rows.join('')}`;
      if (terrainRef.current?.key !== key) {
        terrainRef.current = { key, canvas: buildTerrain(st) };
      }
      ctx.drawImage(terrainRef.current.canvas, 0, 0, W, H);

      const torches = torchSpots(st);
      const lights: { x: number; y: number; r: number; a: number }[] = [];

      // the way out breathes cold light up the stairwell
      for (let ey = 0; ey < height; ey++) {
        for (let ex = 0; ex < width; ex++) {
          if ((st.map.rows[ey]?.[ex] ?? '#') !== 'E') continue;
          const cx = ex * TILE + TILE / 2;
          const cy = ey * TILE + TILE / 2;
          const beat = 0.5 + 0.5 * Math.sin(now / 700);
          lights.push({ x: cx, y: cy, r: TILE * (1.7 + beat * 0.25), a: 0.85 });
          ctx.globalCompositeOperation = 'lighter';
          const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, TILE * 1.1);
          g.addColorStop(0, `rgba(150,205,255,${0.14 + beat * 0.1})`);
          g.addColorStop(1, 'rgba(120,180,255,0)');
          ctx.fillStyle = g;
          ctx.fillRect(cx - TILE * 1.1, cy - TILE * 1.1, TILE * 2.2, TILE * 2.2);
          ctx.globalCompositeOperation = 'source-over';
        }
      }

      // torch flame — two lobes on out-of-phase sine so it wavers
      for (const t of torches) {
        const seed = hash(t.x, t.y, 19) * 100;
        const f = Math.sin(now / 90 + seed) * 0.5 + Math.sin(now / 37 + seed * 2) * 0.5;
        const cx = t.x * TILE + TILE / 2;
        const cy = t.y * TILE + TILE * 0.62;
        const h = 13 + f * 2.5;
        ctx.fillStyle = '#e2611f';
        ctx.beginPath();
        ctx.ellipse(cx + f * 1.2, cy - h / 2, 5, h / 2, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#f6b13a';
        ctx.beginPath();
        ctx.ellipse(cx + f, cy - h / 2 - 1, 3, h / 2.6, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#fff0c0';
        ctx.beginPath();
        ctx.ellipse(cx + f * 0.8, cy - h / 2, 1.4, h / 4.5, 0, 0, Math.PI * 2);
        ctx.fill();
        lights.push({ x: cx, y: cy - 6, r: TILE * (2.6 + f * 0.08), a: 0.9 + f * 0.06 });
      }

      // doors
      for (const door of st.entities.door) {
        const px = door.x * TILE;
        const py = door.y * TILE;
        ctx.fillStyle = '#0c0f14';
        ctx.fillRect(px + 2, py + 1, TILE - 4, TILE - 2);
        if (door.open) {
          ctx.fillStyle = '#4a3a26';
          ctx.fillRect(px + 2, py + 1, 6, TILE - 2);
          ctx.fillRect(px + TILE - 8, py + 1, 6, TILE - 2);
        } else {
          for (let i = 0; i < 4; i++) {
            ctx.fillStyle = i % 2 ? '#6b4a2a' : '#7a5631';
            ctx.fillRect(px + 3 + i * ((TILE - 6) / 4), py + 2, (TILE - 6) / 4 - 1, TILE - 4);
          }
          ctx.fillStyle = '#2b323a';
          ctx.fillRect(px + 3, py + 8, TILE - 6, 5);
          ctx.fillRect(px + 3, py + TILE - 14, TILE - 6, 5);
          ctx.fillStyle = '#c8b25a';
          ctx.beginPath();
          ctx.arc(px + TILE - 13, py + TILE / 2, 3, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = '#1a1207';
          ctx.fillRect(px + TILE - 14, py + TILE / 2, 2, 5);
        }
      }

      const shadow = (cx: number, cy: number, w: number) => {
        ctx.fillStyle = 'rgba(0,0,0,0.42)';
        ctx.beginPath();
        ctx.ellipse(cx, cy, w, w * 0.36, 0, 0, Math.PI * 2);
        ctx.fill();
      };

      const blit = (b: Baked, px: number, py: number, face: number, flash: number) => {
        ctx.drawImage(face < 0 ? b.flip : b.art, px, py, TILE, TILE);
        if (flash > 0) {
          ctx.globalAlpha = flash;
          ctx.drawImage(face < 0 ? b.glowFlip : b.glow, px, py, TILE, TILE);
          ctx.globalAlpha = 1;
        }
      };

      // loot: bobs, and throws its own small light
      for (const item of st.entities.item) {
        if (item.held) continue;
        const bob = Math.sin(now / 380 + item.x * 1.7 + item.y) * 3;
        const px = item.x * TILE;
        const py = item.y * TILE + bob;
        shadow(px + TILE / 2, item.y * TILE + TILE - 8, 11 - bob * 0.4);
        const b = item.kind === 'key' ? baked.key : baked.potion;
        const pulse = 0.35 + 0.25 * (Math.sin(now / 300) * 0.5 + 0.5);
        ctx.globalCompositeOperation = 'lighter';
        ctx.globalAlpha = pulse * 0.5;
        ctx.drawImage(b.glow, px - 2, py - 2, TILE + 4, TILE + 4);
        ctx.globalAlpha = 1;
        ctx.globalCompositeOperation = 'source-over';
        blit(b, px, py, 1, 0);
        lights.push({ x: px + TILE / 2, y: py + TILE / 2, r: TILE * 0.95, a: 0.5 });
      }

      const posOf = (id: string, fx: number, fy: number) => {
        const m = motionRef.current.get(id);
        if (!m) return { x: fx, y: fy, face: 1, flash: 0 };
        const p = Math.min(1, (now - m.t0) / MOVE_MS);
        const e = p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2;
        return {
          x: m.fromX + (m.x - m.fromX) * e,
          y: m.fromY + (m.y - m.fromY) * e,
          face: m.face,
          flash: Math.max(0, 1 - (now - m.hitAt) / FLASH_MS),
        };
      };

      for (const enemy of st.entities.enemy) {
        if (enemy.hp <= 0) continue;
        const m = posOf(enemy.id, enemy.x, enemy.y);
        const bob = Math.sin(now / 240 + enemy.x * 2) * 1.5;
        const px = m.x * TILE;
        const py = m.y * TILE + bob;
        shadow(px + TILE / 2, m.y * TILE + TILE - 6, 12);
        blit(baked.goblin, px, py, m.face, m.flash);
        // hp pips
        const max = 3;
        for (let i = 0; i < max; i++) {
          ctx.fillStyle = i < enemy.hp ? '#d8443f' : 'rgba(255,255,255,0.18)';
          ctx.fillRect(px + TILE / 2 - 9 + i * 7, py + 4, 5, 3);
        }
      }

      const p = player(st);
      const pm = posOf(p.id, p.x, p.y);
      const hbob = Math.sin(now / 300) * 1.2;
      const ppx = pm.x * TILE;
      const ppy = pm.y * TILE + hbob;
      shadow(ppx + TILE / 2, pm.y * TILE + TILE - 6, 13);
      blit(baked.hero, ppx, ppy, pm.face, pm.flash);
      lights.push({ x: ppx + TILE / 2, y: ppy + TILE / 2, r: TILE * 1.5, a: 0.6 });

      // ---- lighting: one dark sheet with the lit areas punched out of it
      let dark = darkRef.current;
      if (!dark) {
        dark = document.createElement('canvas');
        darkRef.current = dark;
      }
      if (dark.width !== W || dark.height !== H) {
        dark.width = W;
        dark.height = H;
      }
      const dctx = dark.getContext('2d')!;
      dctx.globalCompositeOperation = 'source-over';
      dctx.clearRect(0, 0, W, H);
      dctx.fillStyle = 'rgba(6,9,14,0.62)';
      dctx.fillRect(0, 0, W, H);
      dctx.globalCompositeOperation = 'destination-out';

      // the model's 5x5 window is the brightest thing on the board
      const v = st.vision;
      const vcx = (pm.x + 0.5) * TILE;
      const vcy = (pm.y + 0.5) * TILE;
      const vr = (v + 0.5) * TILE;
      const vg = dctx.createRadialGradient(vcx, vcy, vr * 0.55, vcx, vcy, vr * 1.32);
      vg.addColorStop(0, 'rgba(0,0,0,1)');
      vg.addColorStop(0.62, 'rgba(0,0,0,0.9)');
      vg.addColorStop(1, 'rgba(0,0,0,0)');
      dctx.fillStyle = vg;
      dctx.fillRect(vcx - vr * 1.4, vcy - vr * 1.4, vr * 2.8, vr * 2.8);

      for (const l of lights) {
        const g = dctx.createRadialGradient(l.x, l.y, 0, l.x, l.y, l.r);
        g.addColorStop(0, `rgba(0,0,0,${l.a})`);
        g.addColorStop(0.55, `rgba(0,0,0,${l.a * 0.4})`);
        g.addColorStop(1, 'rgba(0,0,0,0)');
        dctx.fillStyle = g;
        dctx.fillRect(l.x - l.r, l.y - l.r, l.r * 2, l.r * 2);
      }
      ctx.drawImage(dark, 0, 0, W, H);

      // ---- warm bloom over the lit areas
      ctx.globalCompositeOperation = 'lighter';
      for (const l of lights) {
        const g = ctx.createRadialGradient(l.x, l.y, 0, l.x, l.y, l.r * 0.75);
        g.addColorStop(0, 'rgba(255,176,74,0.16)');
        g.addColorStop(1, 'rgba(255,150,60,0)');
        ctx.fillStyle = g;
        ctx.fillRect(l.x - l.r, l.y - l.r, l.r * 2, l.r * 2);
      }
      ctx.globalCompositeOperation = 'source-over';

      // ---- what the model could see this turn, stated plainly
      const bx = (pm.x - v) * TILE;
      const by = (pm.y - v) * TILE;
      const bs = (2 * v + 1) * TILE;
      ctx.strokeStyle = `rgba(255,226,150,${0.5 + 0.18 * Math.sin(now / 500)})`;
      ctx.lineWidth = 2;
      ctx.setLineDash([7, 5]);
      ctx.lineDashOffset = -now / 60;
      ctx.strokeRect(bx + 1, by + 1, bs - 2, bs - 2);
      ctx.setLineDash([]);
      ctx.lineDashOffset = 0;

      // ---- vignette
      const vig = ctx.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.35, W / 2, H / 2, Math.max(W, H) * 0.72);
      vig.addColorStop(0, 'rgba(0,0,0,0)');
      vig.addColorStop(1, 'rgba(0,0,0,0.55)');
      ctx.fillStyle = vig;
      ctx.fillRect(0, 0, W, H);

      if (st.status !== 'playing') {
        ctx.fillStyle = st.status === 'won' ? 'rgba(52,140,90,0.22)' : 'rgba(150,40,40,0.28)';
        ctx.fillRect(0, 0, W, H);
      }
    };

    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, []);

  return <canvas ref={canvasRef} className="dungeon" />;
}
