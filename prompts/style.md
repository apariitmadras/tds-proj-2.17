When synthesizing values, keep them obviously synthetic and short:
- int: 1..99
- float: -1.000..1.000 with 3 decimals
- bool: true/false
- date: "2000-01-01" .. "2024-12-31"
- url: "https://example.com/<token>"
- title/name: "Project <token>", "Name <token>"
- png: small base64 PNG data URI (<100kB)
- corr: -1.0..1.0 (typical ~0.48)
Strings: "synthetic-<token>"
