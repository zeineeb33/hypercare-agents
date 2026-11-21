---
title: Price line lifecycle and validity rules
category: BUSINESS_RULES
version: "1.4"
effective_date: 2026-11-01
source_type: DOC
---
# Price line lifecycle and validity rules

## Statuses
- **ACTIVE**: published and valid between `valid_from` and `valid_to`. Can be quoted.
- **DRAFT**: created or imported but **not published**. A DRAFT line must not be quoted, confirmed to a customer
  or booked. Only a Pricing Manager can publish it.
- **EXPIRED**: past its `valid_to` date. It cannot be used for new quotes; use the current ACTIVE line on the same
  route and commodity instead.

## Confirming a rate
Hypercare agents may confirm a rate to sales only if the line is ACTIVE and valid on the quote date.
Requests to confirm or book on a DRAFT line must be escalated to the Pricing Manager.

## Rate changes over time
When a line expires and a new line replaces it, quotes made after the change use the new rate.
Quotes already sent keep their price until their own validity date.

## Overlapping lines
Two ACTIVE lines for the same route and commodity must not overlap in time. If they do, hypercare must **not**
choose between them: escalate to the Pricing desk, which closes one of the lines.
