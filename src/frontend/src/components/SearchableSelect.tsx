import { useState, useRef, useEffect, type ReactNode } from 'react';
import { ChevronDown } from 'lucide-react';
import { LoadingSpinner, LoadingSpinnerInset } from './LoadingSpinner';

export interface SelectOption {
  value: string;
  label: string;
  description?: string;
}

export interface SelectGroup {
  label: string;
  options: SelectOption[];
}

interface Props {
  value: string;
  onChange: (value: string) => void;
  /** Flat list of options (use this OR groups, not both) */
  options?: SelectOption[];
  /** Grouped options with section headers */
  groups?: SelectGroup[];
  /** Label for the empty-value option (shown in trigger when nothing selected) */
  placeholder?: string;
  disabled?: boolean;
  /** Extra classes applied to the trigger element */
  className?: string;
  /** Icon rendered on the left side of the trigger */
  leftIcon?: ReactNode;
  /**
   * When true (default), the placeholder is shown as the first selectable item
   * in the dropdown so the user can clear the selection. Set false for selects
   * where value='' is never valid (e.g. experiment picker that's always set).
   */
  allowClear?: boolean;
  /** Called when the dropdown opens — use for lazy-loading options */
  onOpen?: () => void;
  /** When true, shows a loading spinner inside the open dropdown */
  loading?: boolean;
}

export default function SearchableSelect({
  value,
  onChange,
  options = [],
  groups,
  placeholder = 'Select...',
  disabled = false,
  className = '',
  leftIcon,
  allowClear = true,
  onOpen,
  loading = false,
}: Props) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Flatten all options for label lookup
  const allOptions: SelectOption[] = groups ? groups.flatMap((g) => g.options) : options;
  const selectedLabel = allOptions.find((o) => o.value === value)?.label ?? value;

  const q = query.toLowerCase();
  const filter = (opts: SelectOption[]) =>
    q
      ? opts.filter(
          (o) =>
            o.label.toLowerCase().includes(q) ||
            o.value.toLowerCase().includes(q) ||
            (o.description?.toLowerCase().includes(q) ?? false),
        )
      : opts;

  // Close when clicking outside
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
        setQuery('');
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  const handleOpen = () => {
    if (disabled) return;
    onOpen?.();
    setOpen(true);
    setQuery('');
    setTimeout(() => inputRef.current?.focus(), 10);
  };

  const handleSelect = (val: string) => {
    onChange(val);
    setOpen(false);
    setQuery('');
  };

  const renderItems = () => {
    if (groups) {
      const anyResults = groups.some((g) => filter(g.options).length > 0);
      if (!anyResults) {
        return (
          <div className="px-3 py-2 text-sm text-gray-400 text-center">No results</div>
        );
      }
      return groups.map((g) => {
        const filtered = filter(g.options);
        if (filtered.length === 0) return null;
        return (
          <div key={g.label}>
            <div className="px-3 py-1 text-[10px] font-semibold uppercase tracking-wider text-gray-400 bg-gray-50 sticky top-0">
              {g.label}
            </div>
            {filtered.map((o) => (
              <button
                key={o.value}
                type="button"
                onMouseDown={() => handleSelect(o.value)}
                className={`w-full text-left px-3 py-1.5 transition-colors hover:bg-red-50 hover:text-databricks-red ${
                  o.value === value
                    ? 'bg-red-50 text-databricks-red font-medium'
                    : 'text-gray-700'
                }`}
              >
                <span className="block text-sm">{o.label}</span>
                {o.description && (
                  <span className="block text-[10px] text-gray-400 mt-0.5 font-normal">{o.description}</span>
                )}
              </button>
            ))}
          </div>
        );
      });
    }

    const filtered = filter(options);
    if (filtered.length === 0) {
      return (
        <div className="px-3 py-2 text-sm text-gray-400 text-center">No results</div>
      );
    }
    return filtered.map((o) => (
      <button
        key={o.value || '__empty__'}
        type="button"
        onMouseDown={() => handleSelect(o.value)}
        className={`w-full text-left px-3 py-1.5 transition-colors hover:bg-red-50 hover:text-databricks-red ${
          o.value === value
            ? 'bg-red-50 text-databricks-red font-medium'
            : 'text-gray-700'
        }`}
      >
        <span className="block text-sm">{o.label}</span>
        {o.description && (
          <span className="block text-[10px] text-gray-400 mt-0.5 font-normal">{o.description}</span>
        )}
      </button>
    ));
  };

  return (
    <div ref={containerRef} className="relative w-full">
      {/* Trigger */}
      <div
        onClick={handleOpen}
        className={`
          relative w-full rounded-lg border-0 bg-white shadow-sm cursor-pointer
          flex items-center
          ${leftIcon ? 'pl-9' : 'pl-3.5'} pr-8 py-2
          ${open ? 'ring-2 ring-inset ring-databricks-red' : 'ring-1 ring-inset ring-gray-300'}
          ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
          transition-shadow duration-150
          ${className}
        `}
      >
        {leftIcon && (
          <div className="absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none">
            {leftIcon}
          </div>
        )}

        {open ? (
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Escape') {
                setOpen(false);
                setQuery('');
              }
            }}
            onClick={(e) => e.stopPropagation()}
            className="flex-1 bg-transparent outline-none text-sm text-gray-900 min-w-0 placeholder:text-gray-400"
            placeholder="Search..."
          />
        ) : (
          <span
            className={`flex-1 truncate text-sm leading-6 select-none ${
              value ? 'text-gray-900' : 'text-gray-400'
            }`}
          >
            {value ? selectedLabel : placeholder}
          </span>
        )}

        {loading && !open && <LoadingSpinnerInset />}

        <ChevronDown
          className={`absolute right-2.5 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none transition-transform duration-150 ${
            open ? 'rotate-180' : ''
          }`}
        />
      </div>

      {/* Dropdown */}
      {open && (
        <div className="absolute z-50 mt-1 w-full bg-white rounded-lg shadow-lg ring-1 ring-gray-200 overflow-hidden">
          <div className="max-h-52 overflow-y-auto">
            {loading ? (
              <div className="flex items-center gap-2 px-3 py-3 text-xs text-gray-400">
                <LoadingSpinner className="w-3.5 h-3.5 text-gray-400" /> Loading…
              </div>
            ) : (
              <>
                {allowClear && (
                  <button
                    type="button"
                    onMouseDown={() => handleSelect('')}
                    className={`w-full text-left px-3 py-1.5 text-sm transition-colors hover:bg-red-50 hover:text-databricks-red ${
                      !value
                        ? 'text-databricks-red font-medium bg-red-50'
                        : 'text-gray-400'
                    }`}
                  >
                    {placeholder}
                  </button>
                )}
                {renderItems()}
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
