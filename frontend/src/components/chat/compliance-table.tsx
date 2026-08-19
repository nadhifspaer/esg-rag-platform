import type { ComplianceResult } from "@/lib/types";

import { AnswerBody } from "./answer-body";

/** Renders a `/compliance-check` result inline in the chat thread, reusing `AnswerBody`'s markdown-table renderer. */
export function ComplianceTable({
  result,
  messageId,
}: {
  result: ComplianceResult;
  messageId: string;
}) {
  return (
    <div className="space-y-2">
      <p className="text-xs font-medium uppercase tracking-wide text-muted">
        Compliance check · {result.category}
      </p>
      <AnswerBody answer={result.markdown} messageId={messageId} />
    </div>
  );
}
