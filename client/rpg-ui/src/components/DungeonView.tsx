// The dungeon, drawn on a canvas. Everything here is painted from arrays in
// code — no sprite sheets, no third-party or lifted game art.
//
// The human sees the whole map; the box outlines the 5x5 the model was
// actually shown this turn, so it is obvious when a bad move was a bad
// decision rather than a blind one.
import { useEffect, useRef } from 'react';
import { player, type RpgState } from '../lib/rpgApi';

const TILE = 44;

const COLORS = {
  wall: '#2f4033',
  wallTop: '#3d5442',
  floor: '#d8c9a3',
  floorAlt: '#d0c098',
  exit: '#8a8f98',
  exitDark: '#5d626b',
  doorLocked: '#8a5a2b',
  doorOpen: '#c8b48a',
  player: '#2f7d4f',
  playerTrim: '#eae3cf',
  enemy: '#8c3b3b',
  key: '#d8a53a',
  potion: '#b8456a',
  vision: 'rgba(255, 246, 214, 0.10)',
  visionEdge: 'rgba(255, 226, 130, 0.85)',
};

/** A tiny pixel figure, drawn as filled cells on a 7x7 grid inside a tile. */
function pixels(
  ctx: CanvasRenderingContext2D,
  ox: number,
  oy: number,
  rows: string[],
  color: string
) {
  const unit = TILE / 7;
  ctx.fillStyle = color;
  rows.forEach((row, ry) => {
    for (let rx = 0; rx < row.length; rx++) {
      if (row[rx] !== ' ') ctx.fillRect(ox + rx * unit, oy + ry * unit, unit + 0.5, unit + 0.5);
    }
  });
}

const HERO = ['  xxx  ', '  xxx  ', ' xxxxx ', 'xxxxxxx', ' xx xx ', ' xx xx ', ' x   x '];
const GOBLIN = [' x   x ', ' xxxxx ', 'xx x xx', 'xxxxxxx', ' xxxxx ', ' x x x ', ' x   x '];
const KEY = ['  xxx  ', ' xx xx ', ' xx xx ', '  xxx  ', '   x   ', '   xx  ', '   xx  '];
const POTION = ['  xxx  ', '   x   ', '  xxx  ', ' xxxxx ', ' xxxxx ', ' xxxxx ', '  xxx  '];

interface Props {
  state: RpgState;
}

export function DungeonView({ state }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { width, height, rows } = state.map;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * TILE * dpr;
    canvas.height = height * TILE * dpr;
    canvas.style.width = `${width * TILE}px`;
    canvas.style.height = `${height * TILE}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.imageSmoothingEnabled = false;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const tile = rows[y]?.[x] ?? '#';
        const px = x * TILE;
        const py = y * TILE;
        if (tile === '#') {
          ctx.fillStyle = COLORS.wall;
          ctx.fillRect(px, py, TILE, TILE);
          ctx.fillStyle = COLORS.wallTop;
          ctx.fillRect(px, py, TILE, TILE * 0.22);
        } else {
          ctx.fillStyle = (x + y) % 2 ? COLORS.floorAlt : COLORS.floor;
          ctx.fillRect(px, py, TILE, TILE);
        }
        if (tile === 'E') {
          // stairs down: receding bands
          for (let i = 0; i < 4; i++) {
            ctx.fillStyle = i % 2 ? COLORS.exit : COLORS.exitDark;
            ctx.fillRect(px + 4 + i * 3, py + 6 + i * 8, TILE - 8 - i * 6, 7);
          }
        }
      }
    }

    for (const door of state.entities.door) {
      const px = door.x * TILE;
      const py = door.y * TILE;
      ctx.fillStyle = door.open ? COLORS.doorOpen : COLORS.doorLocked;
      ctx.fillRect(px + 3, py + 2, TILE - 6, TILE - 4);
      if (!door.open) {
        ctx.fillStyle = COLORS.key;
        ctx.fillRect(px + TILE - 14, py + TILE / 2 - 3, 6, 6);
      }
    }

    for (const item of state.entities.item) {
      if (item.held) continue;
      pixels(ctx, item.x * TILE, item.y * TILE,
             item.kind === 'key' ? KEY : POTION,
             item.kind === 'key' ? COLORS.key : COLORS.potion);
    }

    for (const enemy of state.entities.enemy) {
      if (enemy.hp <= 0) continue;
      pixels(ctx, enemy.x * TILE, enemy.y * TILE, GOBLIN, COLORS.enemy);
    }

    const p = player(state);
    pixels(ctx, p.x * TILE, p.y * TILE, HERO, COLORS.player);
    pixels(ctx, p.x * TILE, p.y * TILE, ['       ', '   x   ', '       ', '       ', '       ', '       ', '       '], COLORS.playerTrim);

    // what the model could see this turn
    const v = state.vision;
    const vx = (p.x - v) * TILE;
    const vy = (p.y - v) * TILE;
    const vs = (2 * v + 1) * TILE;
    ctx.fillStyle = COLORS.vision;
    ctx.fillRect(vx, vy, vs, vs);
    ctx.strokeStyle = COLORS.visionEdge;
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(vx + 1, vy + 1, vs - 2, vs - 2);
    ctx.setLineDash([]);
  }, [state]);

  return <canvas ref={canvasRef} className="dungeon" />;
}
