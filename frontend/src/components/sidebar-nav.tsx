import Link from "next/link";

/** The application shell's left-column navigation, present on every route via `layout.tsx`. */
export function SidebarNav() {
  return (
    <nav
      aria-label="Main"
      className="flex w-56 shrink-0 flex-col gap-1 border-r border-border bg-surface px-4 py-6"
    >
      <Link
        href="/about"
        className="mb-6 flex flex-col rounded-md px-2 py-1.5 leading-tight transition-colors hover:bg-surface-raised"
      >
        <span className="text-sm font-semibold tracking-tight text-foreground">
          ESG Intelligence Platform
        </span>
        <span className="text-xs text-muted">Indonesian Banking</span>
      </Link>

      <Link
        href="/chat"
        className="rounded-md px-2 py-1.5 text-sm text-muted transition-colors hover:bg-surface-raised hover:text-foreground"
      >
        Chat
      </Link>

      <Link
        href="/documents"
        className="rounded-md px-2 py-1.5 text-sm text-muted transition-colors hover:bg-surface-raised hover:text-foreground"
      >
        Documents
      </Link>
    </nav>
  );
}
