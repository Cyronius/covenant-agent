"""Scripted players for the decision worlds (family A).

Each module exposes `plan_turn(state, budget) -> authoring program text`,
the same contract `harness/rpg_oracle.py` set: the oracle cheats on
perception (it plans over the whole state, not the rendered window) but not
on mechanics - it emits ordinary Agent Core programs and they go through
resolve -> typecheck -> compile -> sandbox like a model's would. If the
oracle cannot finish a scenario, the scenario is unwinnable and a model
failing it proves nothing.

They are also the labeller: `data/gen/episodes.py` asks the oracle what it
would do *from the state the episode is actually in*, including states a
deliberately wrong move produced.
"""
