/** POST /api/demo-login: signs the visitor into the shared demo account server-side, so the credentials never reach the browser. */

import { NextResponse } from "next/server";

import { createClient } from "@/lib/supabase/server";

export async function POST() {
  const email = process.env.DEMO_USER_EMAIL;
  const password = process.env.DEMO_USER_PASSWORD;

  if (!email || !password) {
    // A configuration fault, not a credential problem: say so plainly rather than
    // reporting it as a failed login, which would send someone hunting the wrong bug.
    return NextResponse.json(
      {
        error:
          "Demo login is not configured on this server. Set DEMO_USER_EMAIL and " +
          "DEMO_USER_PASSWORD (see .env.local.example).",
      },
      { status: 503 },
    );
  }

  const supabase = createClient();
  const { data, error } = await supabase.auth.signInWithPassword({ email, password });

  if (error) {
    // The Supabase message is logged for the operator but not returned: on a shared demo
    // account, "wrong password" is information about our configuration, not about the caller.
    console.error("demo login failed:", error.message);
    return NextResponse.json(
      { error: "Could not sign in to the demo account. Check the server configuration." },
      { status: 502 },
    );
  }

  // The session is already set as cookies by the server client. Return only what the UI needs.
  return NextResponse.json({ email: data.user?.email ?? null });
}
