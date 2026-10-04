interface FluidBlobsProps {
  lightColors: string[];
  darkColors?: string[];
  origins: { x: number; y: number }[]; // percent of the box; negative y sits above it
  margin?: number; // px each blob extends past the box
  blur?: number; // px
}

/** Slow, blurred colour fields drifting inside their box. Light theme colours are used. */
export function FluidBlobs({ lightColors, origins, margin = 60, blur = 50 }: FluidBlobsProps) {
  return (
    <div aria-hidden="true" className="pointer-events-none absolute inset-0 overflow-hidden">
      {lightColors.map((color, i) => {
        const o = origins[i % origins.length];
        return (
          <span
            key={i}
            className="fluid-blob"
            style={
              {
                left: `calc(${o.x}% - 50% - ${margin}px)`,
                top: `calc(${o.y}% - ${margin}px)`,
                width: `calc(100% + ${margin * 2}px)`,
                height: `calc(100% + ${margin * 2}px)`,
                background: `radial-gradient(closest-side, ${color}, transparent)`,
                filter: `blur(${blur}px)`,
                animationDuration: `${9 + i * 3}s`,
                animationDelay: `${-i * 2.5}s`,
                "--blob-dx": `${i % 2 ? -1 : 1}`,
              } as React.CSSProperties
            }
          />
        );
      })}
    </div>
  );
}
