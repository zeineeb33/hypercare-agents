---
title: Base rate, surcharges and all-in prices
category: BUSINESS_RULES
version: "3.0"
effective_date: 2026-10-01
source_type: DOC
---
# Base rate, surcharges and all-in prices

## Base rate
The rate on a price line (20ST, 40ST, 40HC) is the **base ocean freight**.

## Surcharges
Surcharges are added per container on top of the base rate:
- **BAF** (bunker adjustment factor) covers fuel price changes.
- **LSS** (low sulphur surcharge) covers the extra cost of low-sulphur fuel required by emission rules.
- **CAF** (currency adjustment factor) covers exchange-rate changes.
- **PSS** (peak season surcharge) applies during announced peak periods.
- **HAZ** applies to hazardous cargo only.

## All-in price lists
Some price lists are published **all-in** (flag `charges_included`): their rates already include every surcharge,
and no surcharge is added on top.

## What the customer sees
QuotePort always shows the customer an **all-in price per container**: base rate plus the surcharges of the line.
A quote can therefore be higher than the rate displayed on the price line. Example: base 40HC rate 1,850 USD,
BAF 220 USD and LSS 80 USD give a quoted all-in price of 2,150 USD.
