/** Middleware: refreshes the Supabase session on every request and auto-signs in a visitor with no session. */

import { createServerClient, type CookieOptions } from "@supabase/ssr";
import { NextResponse, type NextRequest } from "next/server";

export async function middleware(request: NextRequest) {
  let response = NextResponse.next({ request: { headers: request.headers } });

  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        get(name: string) {
          return request.cookies.get(name)?.value;
        },
        set(name: string, value: string, options: CookieOptions) {
          request.cookies.set({ name, value, ...options });
          response = NextResponse.next({ request: { headers: request.headers } });
          response.cookies.set({ name, value, ...options });
        },
        remove(name: string, options: CookieOptions) {
          request.cookies.set({ name, value: "", ...options });
          response = NextResponse.next({ request: { headers: request.headers } });
          response.cookies.set({ name, value: "", ...options });
        },
      },
    },
  );

  // The call itself is the point: it triggers the refresh-and-rewrite above.
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    const email = process.env.DEMO_USER_EMAIL;
    const password = process.env.DEMO_USER_PASSWORD;
    if (email && password) {
      // Errors deliberately not logged here: this runs on every unauthenticated request.
      await supabase.auth.signInWithPassword({ email, password });
    }
  }

  return response;
}

export const config = {
  // Skip static assets and image optimisation: refreshing a session on a favicon request is
  // wasted work, and on this project it would also be a wasted outbound call to Supabase.
  matcher: ["/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp|woff2?)$).*)"],
};
