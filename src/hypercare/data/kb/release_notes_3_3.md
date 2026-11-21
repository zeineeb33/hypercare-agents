---
title: TideRate 3.3 release notes
category: RELEASE_NOTES
version: "3.3"
effective_date: 2026-11-01
source_type: DOC
---
# TideRate 3.3 release notes

- **DRAFT status**: lines imported from the pricing workbench now start as DRAFT and must be published by a
  Pricing Manager before they can be quoted.
- **Blocking quotes on DRAFT lines**: QuotePort no longer creates quotes on DRAFT lines. Quotes created on such lines
  before the upgrade may still exist and must not be booked; escalate them to the Pricing Manager.
- **Dynamic pricing (beta)**: available on the North Europe to Asia trade only. See the demo video.
- Fix: the inland tariff table is filtered by gateway port.
