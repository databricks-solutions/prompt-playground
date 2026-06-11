import { Loader2 } from 'lucide-react';

type SizeProps = { className?: string };

/**
 * Lucide loader ring — only `rotate` is applied here. Never add `-translate-y-*` on the same
 * node as `animate-spin`; wrap in a parent that handles vertical centering instead.
 */
export function LoadingSpinner({ className = 'w-4 h-4 text-current' }: SizeProps) {
  return (
    <Loader2
      className={`${className} inline-block origin-center animate-spin motion-reduce:animate-none`}
      aria-hidden
    />
  );
}

/** Vertically centered strip icon on controls — outer span has translate-y; inner spins cleanly */
export function LoadingSpinnerInset({
  className = 'w-3.5 h-3.5 text-amber-600',
  rightClassName = 'right-8',
}: SizeProps & { rightClassName?: string }) {
  return (
    <span
      className={`pointer-events-none absolute ${rightClassName} top-1/2 flex -translate-y-1/2 items-center justify-center`}
      aria-hidden
    >
      <LoadingSpinner className={className} />
    </span>
  );
}
