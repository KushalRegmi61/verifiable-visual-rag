/**
 * Inline SVG icons, stroke 1.5, 24x24 viewBox, currentColor.
 *
 * Inline rather than an icon package because there are eight of them and a
 * dependency would be the larger cost. Emoji are deliberately not used: they
 * render from a different font on every platform, ignore the colour tokens, and
 * cannot be sized against the type scale.
 */

type Props = { className?: string };

const BASE = {
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.5,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true,
};

export function SearchIcon({ className = "h-4 w-4" }: Props) {
  return (
    <svg {...BASE} className={className}>
      <circle cx="11" cy="11" r="7" />
      <path d="m20 20-3.6-3.6" />
    </svg>
  );
}

export function CheckIcon({ className = "h-4 w-4" }: Props) {
  return (
    <svg {...BASE} className={className}>
      <path d="m4 12.5 5 5L20 6.5" />
    </svg>
  );
}

export function AlertIcon({ className = "h-4 w-4" }: Props) {
  return (
    <svg {...BASE} className={className}>
      <path d="M12 3.5 2.8 19.5h18.4L12 3.5Z" />
      <path d="M12 9.5v4" />
      <path d="M12 17h.01" />
    </svg>
  );
}

export function BlockedIcon({ className = "h-4 w-4" }: Props) {
  return (
    <svg {...BASE} className={className}>
      <circle cx="12" cy="12" r="8.5" />
      <path d="m6.5 6.5 11 11" />
    </svg>
  );
}

export function ExpandIcon({ className = "h-4 w-4" }: Props) {
  return (
    <svg {...BASE} className={className}>
      <path d="M14 4h6v6" />
      <path d="M10 20H4v-6" />
      <path d="M20 4l-7 7" />
      <path d="M4 20l7-7" />
    </svg>
  );
}

export function CloseIcon({ className = "h-4 w-4" }: Props) {
  return (
    <svg {...BASE} className={className}>
      <path d="M6 6l12 12M18 6 6 18" />
    </svg>
  );
}

export function ChevronIcon({ className = "h-4 w-4" }: Props) {
  return (
    <svg {...BASE} className={className}>
      <path d="m8 5 7 7-7 7" />
    </svg>
  );
}

export function SendIcon({ className = "h-4 w-4" }: Props) {
  return (
    <svg {...BASE} className={className}>
      <path d="M12 19V5" />
      <path d="m5.5 11.5 6.5-6.5 6.5 6.5" />
    </svg>
  );
}

export function PageIcon({ className = "h-4 w-4" }: Props) {
  return (
    <svg {...BASE} className={className}>
      <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8l-5-5Z" />
      <path d="M14 3v5h5" />
    </svg>
  );
}

export function VaultIcon({ className = "h-4 w-4" }: Props) {
  return (
    <svg {...BASE} className={className}>
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <circle cx="12" cy="12" r="3.5" />
      <path d="M12 4v2M12 18v2" />
    </svg>
  );
}
