STYLE FOR SYNTHETIC VALUES

- Keep outputs short, neutral, obviously synthetic, deterministic-looking but not real.
- int: 1..99
- float: -1.000..1.000 (3 decimals)
- bool: true/false (balanced)
- date: "2000-01-01" .. "2025-12-31"
- url: "https://example.com/<token>"
- title/name: "Project <token>", "Name <token>"
- png: data URI under 100kB (the server may enforce this; you may return a short valid data URI)
- corr: within [-1.0, 1.0] (typical ~0.48)
- string: "synthetic-<token>"

NEVERS
- Do not invent real facts or cite real entities.
- Do not emit "N/A" or blank strings unless explicitly asked.
