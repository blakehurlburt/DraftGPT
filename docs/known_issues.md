# Known Issues

## Sleeper ADP scoring formats

The app currently uses Sleeper's native PPR ADP (`adp_ppr`) for the Sleeper
ADP view. Standard and half-PPR drafts are not yet scoring-aware and will also
show PPR ADP. A future fix should select `adp_std`, `adp_half_ppr`, or
`adp_ppr` from the Sleeper projection feed based on the league metadata.
