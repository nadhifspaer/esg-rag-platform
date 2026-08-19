"use client";

/** The three checklist categories the picker below offers. */
export const CATEGORY_OPTIONS: { value: "environmental" | "social" | "governance"; label: string }[] = [
  { value: "environmental", label: "Environmental" },
  { value: "social", label: "Social" },
  { value: "governance", label: "Governance" },
];

/** The frameworks selectable from this control; `"gri"` is first since it's the API's default. */
export const FRAMEWORK_OPTIONS: { value: "gri" | "edgb"; label: string }[] = [
  { value: "gri", label: "GRI" },
  { value: "edgb", label: "EDGB" },
];

/** The framework toggle + E/S/G category buttons; picking a category fires `onPick` immediately. */
export function CategoryFrameworkPicker({
  framework,
  onFrameworkChange,
  onPick,
  disabled,
}: {
  framework: "gri" | "edgb";
  onFrameworkChange: (framework: "gri" | "edgb") => void;
  onPick: (category: "environmental" | "social" | "governance") => void;
  disabled?: boolean;
}) {
  return (
    <>
      {/* A segmented control (not pill buttons) marks this as a closed-set pick, not a topic. */}
      <div className="mt-2 flex items-center gap-2">
        <span className="text-xs text-muted">Framework</span>
        <div className="inline-flex overflow-hidden rounded border border-border">
          {FRAMEWORK_OPTIONS.map((option, i) => (
            <button
              key={option.value}
              type="button"
              onClick={() => onFrameworkChange(option.value)}
              aria-pressed={framework === option.value}
              disabled={disabled}
              className={[
                "px-2 py-1 text-xs transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-accent disabled:opacity-40",
                i > 0 ? "border-l border-border" : "",
                framework === option.value
                  ? "bg-accent text-accent-foreground"
                  : "text-muted hover:bg-surface hover:text-foreground",
              ].join(" ")}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-2 flex flex-wrap gap-1.5">
        {CATEGORY_OPTIONS.map((option) => (
          <button
            key={option.value}
            type="button"
            onClick={() => onPick(option.value)}
            disabled={disabled}
            className="rounded border border-border px-2 py-1 text-xs text-foreground transition-colors hover:bg-surface focus:outline-none focus-visible:ring-2 focus-visible:ring-accent disabled:opacity-40"
          >
            {option.label}
          </button>
        ))}
      </div>
    </>
  );
}
