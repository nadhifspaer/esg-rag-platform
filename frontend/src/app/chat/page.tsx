import type { Metadata } from "next";

import { ChatPanel } from "@/components/chat/chat-panel";
import { createClient } from "@/lib/supabase/server";

export const metadata: Metadata = {
  title: "Chat",
};

/** The chat surface: a Server Component that checks the session before rendering; UX only, not the security boundary. */
export default async function ChatPage() {
  const {
    data: { user },
  } = await createClient().auth.getUser();

  if (!user) {
    return (
      <div className="mx-auto max-w-md px-6 py-20 text-center">
        <h1 className="text-lg font-semibold tracking-tight">Couldn&rsquo;t sign in automatically</h1>
        <p className="mt-2 text-sm text-muted">
          This demo runs on a shared account that normally signs itself in before this page
          loads. That didn&rsquo;t happen this time.
        </p>
        <p className="mt-4 text-xs text-muted">Reloading the page usually resolves it.</p>
      </div>
    );
  }

  return <ChatPanel />;
}
