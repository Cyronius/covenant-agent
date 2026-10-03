# tau2-bench: customer service under a written policy

Borrowed for the `service_retail`, `service_airline` and `service_telecom`
worlds (`.claude/plans/borrowed-worlds.md`, kind 1).

| | |
|---|---|
| Source | https://github.com/sierra-research/tau2-bench |
| Licence | MIT, Copyright (c) 2025 Sierra Research. The full text is in `LICENSE` beside this file and must travel with any copy of these files. |
| Commit | `5bfa7e37b36656b37dc6d022156be6563c1007f3` (cloned 2026-09-29) |
| Split used | tau2's official **train** split only (`split_tasks.json` → `train`) |
| Attribution line | Task scenarios, tool names and descriptions, policies and database records adapted from tau2-bench (Sierra Research, MIT). |

tau-bench (github.com/sierra-research/tau-bench, MIT) is tau2's predecessor.
Its retail and airline domains are carried forward in tau2 and nothing was
taken from it directly.

## What is here

Written by `python -m data.gen.service --extract PATH/TO/tau2-bench`
(`data/gen/service.py`, `extract`). `SOURCE.json` records the commit and the
upstream files each domain came from.

| File | From upstream | What was kept |
|---|---|---|
| `LICENSE` | `LICENSE` | verbatim |
| `retail/tasks_train.json` | `data/tau2/domains/retail/tasks.json` | the 74 train-split tasks, verbatim |
| `retail/policy.md` | `data/tau2/domains/retail/policy.md` | verbatim |
| `retail/db_extract.json` | `data/tau2/domains/retail/db.json` | all 50 products; the users (and their orders) the train tasks touch, plus 220 other users drawn for templates |
| `airline/tasks_train.json` | `data/tau2/domains/airline/tasks.json` | the 30 train-split tasks, verbatim |
| `airline/policy.md` | `data/tau2/domains/airline/policy.md` | verbatim |
| `airline/db_extract.json` | `data/tau2/domains/airline/db.json` | the train tasks' users plus 220 template users, their reservations, and every flight with only status and prices per date (seat counts and actual times dropped) |
| `telecom/tasks_train.json` | `data/tau2/domains/telecom/tasks.json` | the 74 train-split tasks, verbatim |
| `telecom/policy.md` | `data/tau2/domains/telecom/main_policy.md` | verbatim |
| `telecom/db.toml` | `data/tau2/domains/telecom/db.toml` | verbatim |
| `*/denylist.json` | tau2's **test** split | ids only; see below |

**Test split.** No test task is copied. At extraction the test split is read
once, only to list the users whose records its tasks touch (31 retail, 17
airline). Those users are written to `denylist.json` and never drawn for a
template row, so a template row can never re-pose a test scenario. A user who
appears in both splits keeps the train task's rows (the train task is ours to
use) and is still never drawn for templates.

## What was adapted, and how

Our code: `runtime/worlds/service.py` (the three worlds),
`runtime/engines/service.js` (the policy rules), `data/gen/service.py` (the
rows). Every row's `provenance` names `source`, `source_commit`, `licence`,
the upstream `file`, and for converted tasks the `tau2_task` id.

- **Entities** are a subset of tau2's database models, typed in our IR: retail
  user, payment method, product, variant, order, order item; airline user,
  payment method, reservation, segment, dated flight, passenger; telecom
  customer, plan, line, bill, and the customer's phone (tau2's user-side
  device state). Money is INT cents.
- **Tools** keep tau2's names and adapt its docstrings. Where tau2 takes a list
  (`item_ids`, `passengers`, `flights`) our tool takes one record and is
  called once per record, because the language has no list literals. tau2's
  `update_reservation_flights` is split in two: `update_reservation_cabin`
  (same flights, new cabin) and `update_reservation_flights` (one segment to
  another flight). `cancel_reservation` takes the cancellation reason, which
  tau2's policy asks the agent to collect but its API does not take.
  `send_certificate` takes the reservation rather than the user, so the rule
  that decides it can be checked. Telecom's user-side tools
  (`toggle_airplane_mode`, `reseat_sim_card`, ...) are "walk the customer
  through" tools on the phone record. `transfer_to_human_agents`,
  `calculate`, `think`, `book_reservation`, `get_id` and the plan-change flow
  have no tool here.
- **Policy.** tau2 states its policy in prose and leaves most of it to the
  agent ("The API does not check these for the agent"). Here the engine
  enforces it: a forbidden call fails with `PERMISSION_DENIED` and a message
  that starts `policy:` and names the rule. The rules enforced are the
  written policy's: cancel only a pending order, with one of two reasons;
  modify/return/exchange by order status, once per order (`locked`), same
  product, available variant, gift card balance, refund to the original method
  or a gift card; airline cancellation (24 hours, airline-cancelled flight,
  business, insurance with health or weather), no change once flown, basic
  economy flights unchangeable, bags added never removed with the
  membership/cabin allowance, no certificate for changes, compensation
  eligibility and amounts; telecom refuel at most 2 GB, payment request only
  for an overdue bill and one at a time, no resume with unpaid bills or an
  ended contract, a PIN-locked SIM needs a human.
- **Requests** are not tau2's `reason_for_call` text. Each is a one-line
  first-person brief written from the task's expected actions and the
  database (who the customer is, which order or reservation, what they want),
  within the tiny planner's 128-token budget. The situation (order status,
  balances, availability, how long ago a booking was made, the phone's
  settings) is in the constants' descriptions.
- **References** are tau2's expected write actions mapped onto our tools,
  with a read where the program needs one (a user id from email or name and
  zip; "refund to the original payment method" read off the order), and
  re-executed in our sandbox; a row whose reference does not reach its
  expected status is dropped.
- **Refusals.** tau2's `transfer_to_human_agents` and its refusal tasks
  (labelled by `nl_assertions`) become `ABORT UNSUPPORTED <constant>`. The
  airline refusal tasks are mapped by hand in `AIRLINE_POLICY_TASKS`, retail
  tasks 10 and 50 in `RETAIL_POLICY_TASKS`; each mapped refusal is kept only
  if our engine refuses the would-be program too. Telecom's eight
  transfer-only tasks (PIN-locked SIM, ended contract) abort on the phone or
  the line.
- **Variants.** Besides the converted tasks, the generator writes template
  requests of the same kinds over other database records, policy-forbidden
  twins of permitted requests (the refusal confirmed by the engine), and
  ask/answer pairs (`ABORT NEEDS_INFO` for a missing cancellation reason or
  payment method, then the answered request).

## Train tasks that produced no row

- retail 24, 25, 57, 67: no write actions (information requests or a
  change the customer takes back). 105: its brief (two identical kettles,
  each exchanged for a different option) runs over the 128-token budget.
- airline 3, 4, 34, 38: information or negotiation tasks with no write and
  no refusal we could map to one request; 14, 20, 23: `book_reservation`
  (booking is not modelled); 15, 33: a flight change that alters the number
  of segments or the cabin at the same time.
- airline 39's third cancellation (MSJ4OA: economy, insured, 10 days old,
  reason not health or weather) is refused by our engine although tau2's
  label cancels it; that write and the whole-task row are dropped, the other
  two cancellations are kept.

## Regenerating

```
python -m data.gen.service --domain retail --out data/service_retail.jsonl \
    --limit 400 --symbols typed --enums --kinds --decoys 1:2
python -m data.gen.service --domain airline --out data/service_airline.jsonl \
    --limit 400 --symbols typed --enums --kinds --decoys 1:2
python -m data.gen.service --domain telecom --holdout \
    --out data/holdout/e_service_telecom.jsonl --symbols typed --enums --kinds
```

`--decoys` draws the description-only siblings authored in
`runtime/worlds/service.py` (`AUTHORED_DECOYS`). Without them 71.5% of
retail and 100% of airline reference calls are signature-unique; with them
3.6% and 0%. The telecom exam needs none (31.5%: its phone tools share one
signature).
