"""F4 backtranslation data generator.

  random world state
    -> random valid Agent Core program (grammar-guided, level-targeted)
    -> execute -> expected final state
    -> English request (teacher hook; template renderer by default)
    -> training pair (request, tool schemas, program, expected state,
       level, effects), with provenance.

CLI: python -m data.gen --level 3 --n 10000 --seed 42 --out data/L3.jsonl
"""
