import { isIndonesian, parseAnswer, splitMarkers, tableToTsv } from "@/lib/answer";

import { CopyButton } from "./copy-button";

/** Renders an answer as structured content (tables, linked `[n]` citation markers, and lang-tagged blocks) rather than one pre-wrapped paragraph. */
export function AnswerBody({ answer, messageId }: { answer: string; messageId: string }) {
  const blocks = parseAnswer(answer);

  return (
    <div className="space-y-3 text-sm leading-relaxed text-foreground">
      {blocks.map((block, i) =>
        block.kind === "paragraph" ? (
          <p key={i} lang={isIndonesian(block.text) ? "id" : undefined}>
            {splitMarkers(block.text).map((segment, j) =>
              segment.kind === "text" ? (
                segment.text
              ) : (
                <a
                  key={j}
                  href={`#${citationId(messageId, segment.index)}`}
                  // `sup`, not inline brackets: keeps a text selection from capturing "[1]" too.
                  className="ml-0.5 rounded align-super text-[0.65rem] font-medium text-accent underline-offset-2 hover:underline focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                  aria-label={`Source ${segment.index}`}
                >
                  {segment.index}
                </a>
              ),
            )}
          </p>
        ) : (
          <figure key={i} className="space-y-1">
            {/* Wide disclosure tables must scroll inside their own box rather than pushing the
                thread sideways. */}
            <div className="overflow-x-auto rounded-md border border-border">
              <table className="w-full border-collapse text-xs">
                {block.head ? (
                  <thead>
                    <tr className="bg-surface">
                      {block.head.map((cell, c) => (
                        <th
                          key={c}
                          scope="col"
                          className="border-b border-border px-2.5 py-1.5 text-left font-medium text-foreground"
                          lang={isIndonesian(cell) ? "id" : undefined}
                        >
                          {cell}
                        </th>
                      ))}
                    </tr>
                  </thead>
                ) : null}
                <tbody>
                  {block.rows.map((row, r) => (
                    <tr key={r} className="border-b border-border last:border-b-0">
                      {row.map((cell, c) => (
                        <td
                          key={c}
                          // Numeric cells get a tabular, monospaced treatment; other cells don't.
                          className={`px-2.5 py-1.5 align-top ${
                            isNumeric(cell)
                              ? "whitespace-nowrap font-mono tabular-nums text-foreground"
                              : "text-foreground/90"
                          }`}
                          lang={isIndonesian(cell) ? "id" : undefined}
                        >
                          {cell}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <figcaption className="flex justify-end">
              <CopyButton
                text={tableToTsv(block)}
                label="Copy table"
                // TSV so it lands in a spreadsheet as cells rather than as one string. This is
                // the analyst's actual last mile.
                copiedLabel="Copied as TSV"
              />
            </figcaption>
          </figure>
        ),
      )}
    </div>
  );
}

/** Stable id linking an inline marker to its Sources entry, scoped per message. */
export function citationId(messageId: string, index: number): string {
  return `cite-${messageId}-${index}`;
}

/** Values kept verbatim from the corpus use a comma decimal mark and dot thousands. */
function isNumeric(cell: string): boolean {
  return /^[\d.,]+$/.test(cell.trim()) && /\d/.test(cell);
}
