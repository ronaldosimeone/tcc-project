// ── Voto relevante / não relevante — extraído de suggestion-feedback.tsx
// (RNF-58). Clicar de novo no voto marcado o desfaz.

import { ThumbsDown, ThumbsUp } from "lucide-react";
import { Button } from "@/components/ui/button";

interface VoteToggleProps {
  label: string;
  value: boolean | undefined;
  onChange: (value: boolean | undefined) => void;
  disabled: boolean;
}

export const VoteToggle = ({
  label,
  value,
  onChange,
  disabled,
}: VoteToggleProps) => (
  <div className="flex items-center gap-1" role="group" aria-label={label}>
    <Button
      type="button"
      size="icon"
      variant={value === true ? "default" : "outline"}
      className="h-7 w-7"
      aria-pressed={value === true}
      aria-label={`${label}: relevante`}
      disabled={disabled}
      onClick={() => onChange(value === true ? undefined : true)}
    >
      <ThumbsUp className="h-3.5 w-3.5" />
    </Button>
    <Button
      type="button"
      size="icon"
      variant={value === false ? "destructive" : "outline"}
      className="h-7 w-7"
      aria-pressed={value === false}
      aria-label={`${label}: não relevante`}
      disabled={disabled}
      onClick={() => onChange(value === false ? undefined : false)}
    >
      <ThumbsDown className="h-3.5 w-3.5" />
    </Button>
  </div>
);
