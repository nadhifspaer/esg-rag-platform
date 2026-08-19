"use client";

import { useEffect, useState } from "react";

/** A ticking seconds counter for the wait state, so liveness survives `prefers-reduced-motion`. */
export function Elapsed({ startedAt }: { startedAt: number }) {
  const [seconds, setSeconds] = useState(() => Math.floor((Date.now() - startedAt) / 1000));

  useEffect(() => {
    const tick = () => setSeconds(Math.floor((Date.now() - startedAt) / 1000));
    tick();
    const id = window.setInterval(tick, 1000);
    return () => window.clearInterval(id);
  }, [startedAt]);

  return (
    <span className="tabular-nums" aria-hidden="true">
      {seconds}s
    </span>
  );
}
