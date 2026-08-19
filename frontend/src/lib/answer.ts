/** Answer parsing: turning a model answer into renderable blocks (tables vs. paragraphs), recognising only structure that's literally present. */

/** One renderable piece of an answer. */
export type AnswerBlock =
  | { kind: "paragraph"; text: string }
  | { kind: "table"; head: string[] | null; rows: string[][] };

/** A run of answer text, split so inline `[n]` citation markers can become real links. */
export type TextSegment =
  | { kind: "text"; text: string }
  | { kind: "marker"; index: number };

const MARKER_RE = /\[(\d{1,3})\]/g;

/** Bahasa Indonesia function words, used to decide whether a block should carry `lang="id"`. */
const ID_MARKERS = new Set([
  "yang", "dan", "dengan", "dari", "untuk", "pada", "tidak", "adalah", "dalam", "ini",
  "itu", "atau", "akan", "telah", "oleh", "sebagai", "juga", "dapat", "ada", "tahun",
  "jumlah", "total", "laporan", "perusahaan", "kebijakan", "terhadap", "sesuai", "serta",
  "dihasilkan", "limbah", "emisi", "energi", "air", "keberlanjutan", "lingkungan",
]);

/** True when a block is confidently Bahasa Indonesia. See `ID_MARKERS` for why it is strict. */
export function isIndonesian(text: string): boolean {
  const words = text.toLowerCase().match(/[a-z]+/g);
  if (!words || words.length === 0) return false;
  const hits = new Set(words.filter((w) => ID_MARKERS.has(w)));
  if (hits.size >= 2) return true;
  // A short fragment ("Limbah yang dihasilkan") cannot reach two distinct markers, so allow
  // one hit when the fragment is short enough that a single function word is strong evidence.
  return hits.size === 1 && words.length <= 5;
}

/** Split text so inline `[n]` markers can be rendered as links to the Sources list. */
export function splitMarkers(text: string): TextSegment[] {
  const segments: TextSegment[] = [];
  // A fresh regex per call: `exec` advances `lastIndex`, so a shared instance would leak state.
  const scanner = new RegExp(MARKER_RE.source, "g");
  let cursor = 0;
  let match: RegExpExecArray | null;

  while ((match = scanner.exec(text)) !== null) {
    const at = match.index;
    if (at > cursor) segments.push({ kind: "text", text: text.slice(cursor, at) });
    segments.push({ kind: "marker", index: Number(match[1]) });
    cursor = at + match[0].length;
  }
  if (cursor < text.length) segments.push({ kind: "text", text: text.slice(cursor) });
  return segments;
}

function isTableRow(line: string): boolean {
  const trimmed = line.trim();
  return trimmed.startsWith("|") && trimmed.slice(1).includes("|");
}

function isSeparatorRow(cells: string[]): boolean {
  return cells.length > 0 && cells.every((c) => /^:?-{2,}:?$/.test(c.trim()));
}

function cellsOf(line: string): string[] {
  const trimmed = line.trim().replace(/^\|/, "").replace(/\|$/, "");
  return trimmed.split("|").map((c) => c.trim());
}

/** Parse an answer into paragraphs and tables, splitting consecutive pipe-delimited lines out. */
export function parseAnswer(answer: string): AnswerBlock[] {
  const blocks: AnswerBlock[] = [];
  let paragraph: string[] = [];
  let table: string[][] = [];

  const flushParagraph = () => {
    const text = paragraph.join(" ").trim();
    if (text) blocks.push({ kind: "paragraph", text });
    paragraph = [];
  };

  const flushTable = () => {
    if (table.length === 0) return;
    let head: string[] | null = null;
    let rows = table;
    if (table.length >= 2 && isSeparatorRow(table[1])) {
      head = table[0];
      rows = table.slice(2);
    }
    blocks.push({ kind: "table", head, rows });
    table = [];
  };

  for (const line of answer.split("\n")) {
    if (isTableRow(line)) {
      flushParagraph();
      table.push(cellsOf(line));
      continue;
    }
    flushTable();
    if (line.trim() === "") flushParagraph();
    else paragraph.push(line.trim());
  }
  flushTable();
  flushParagraph();

  return blocks;
}

/** Render a table block as TSV, so it pastes into a spreadsheet intact. */
export function tableToTsv(block: { head: string[] | null; rows: string[][] }): string {
  const lines = block.head ? [block.head, ...block.rows] : block.rows;
  return lines.map((row) => row.join("\t")).join("\n");
}
