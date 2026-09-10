// ── Chip de valor bruto instantâneo — RNF-58: extraído de AssetRadarChart ───

interface RawValueProps {
  label: string;
  value: number;
  decimals?: number;
  unit: string;
}

export function RawValue({ label, value, decimals = 2, unit }: RawValueProps) {
  return (
    <span className="text-[11px] text-muted-foreground">
      {label}{" "}
      <span className="font-mono font-semibold text-foreground">
        {value.toFixed(decimals)} <span className="font-normal">{unit}</span>
      </span>
    </span>
  );
}
