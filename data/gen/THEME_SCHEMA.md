# Domain theme packs (S0)

A *theme pack* is one JSON file giving a domain's complete vocabulary. The
compiler (`data/gen/domains.py`) turns it into a WORLD (entities, tools,
sandbox impls), a generation PROFILE (filters, actions, phrases for the
level recipes), and a state generator. Structure is fixed by the compiler;
themes supply ONLY names, descriptions, and phrases.

Every theme has two entities: a **parent** (person/org that owns things)
and a **child** (the work item acted on), linked by a reference field.

## Contract (all keys required unless marked optional)

```jsonc
{
  "domain": "vetclinic",              // unique snake_case id
  "desc_style": "clean",              // clean | terse | verbose | sloppy
                                      // — write ALL tool descs in this style

  "parent": {
    "entity": "owner",                // record type (singular snake_case)
    "noun": ["owner", "owners"],
    "name_style": "person",           // person | company (which name bank)
    "contact_field": "email"          // email | phone | null
  },

  "child": {
    "entity": "appointment",
    "noun": ["appointment", "appointments"],
    "ref_field": "owner",             // FK field on child -> parent
    "ref_phrase": "for {name}",       // request phrase for that filter
    "name_field": "reason",           // display field, or null
    "titles": ["Annual checkup", "Vaccination booster", "Limp in left paw",
               "Dental cleaning", "Spay surgery", "Ear infection recheck",
               "Microchip fitting", "Weight consult", "Post-op follow-up",
               "Allergy testing", "Senior wellness exam", "Nail trim"],
                                      // >=12 if name_field set, else []
    "bools": [                        // 1-3 boolean fields
      {"field": "no_show_risk", "true_phrase": "flight-risk",
       "false_phrase": "reliable", "placement": "pre"}
    ],
    "enum": {                         // exactly one enum field, 2-4 values
      "field": "status", "values": ["scheduled", "completed", "cancelled"],
      "phrases": {"scheduled": ["upcoming", "pre"],
                  "completed": ["completed", "pre"],
                  "cancelled": ["cancelled", "pre"]}
    },
    "times": [                        // 1-2 TIME fields
      {"field": "slot", "kind": "time_now",
       "lt_phrase": ["already past", "post"],
       "gt_phrase": ["still ahead", "post"]},
      {"field": "booked", "kind": "time_cutoff",
       "phrase": "booked more than {d} days ago"}
    ],

    "number": {                       // optional: one INT field, so the
                                      // compute tools (sum/avg/max/min)
                                      // have something to be called on
      "field": "duration_minutes",
      "noun": "appointment length",   // how a request says it: "the total
                                      // appointment length across the
                                      // appointments" — NOT the field name
      "min": 10, "max": 90            // per-record range, ints, min < max
    }
  },

  // Tool slots. The compiler fixes each slot's structure/effects/impl;
  // the theme names it and describes it (in desc_style!). "verbs" are
  // imperative request templates ({obj} = the item phrase).
  "tools": {
    "list_child":  {"name": "list_appointments",
                    "desc": "List every appointment on the clinic calendar."},
    "get_child":   {"name": "get_appointment",
                    "desc": "Fetch one appointment by its id."},
    "list_parent": {"name": "list_owners",
                    "desc": "List all pet owners registered with the clinic."},
    "get_parent":  {"name": "get_owner",
                    "desc": "Fetch one pet owner by id."},
    "delete_child": {"name": "purge_appointment",
                     "desc": "Permanently remove an appointment record. Not reversible.",
                     "verbs": ["purge {obj}", "wipe {obj} from the calendar"]},
    "set_enum":    {"name": "set_appointment_status",
                    "desc": "Change an appointment's status (scheduled, completed, or cancelled).",
                    "verbs": ["mark {obj} {to_value}", "set {obj} {to_value}"],
                    "value_phrases": {"scheduled": "back to scheduled",
                                      "completed": "as completed",
                                      "cancelled": "as cancelled"}},
    "set_bool":    {"name": "flag_no_show_risk",
                    "bool_index": 0, "value": true,
                    "desc": "Flag an appointment as a no-show risk.",
                    "verbs": ["flag {obj} as a no-show risk",
                              "mark {obj} flight-risk"]},
    "set_ref":     {"name": "rebook_owner",
                    "desc": "Move an appointment to a different owner's account.",
                    "verbs": ["move {obj} {to_name}", "rebook {obj} {to_name}"],
                    "to_template": "under {name}"},
    "send":        {"name": "text_owner",
                    "desc": "Send an SMS to a pet owner.",
                    "child_verbs": ["text the owner of {obj}",
                                    "give the owner of {obj} a heads-up"],
                    "direct_verbs": ["text {name}", "send {name} an SMS"],
                    "pair_verbs": ["text", "notify"]}
  },

  "sort": {                           // superlative/ordinal recipe material
    "time_index": 0,                  // which child.times field to sort by
    "dir": "ASC",
    "sup_phrase": "the appointment with the earliest slot",
    "ord_phrase": "the appointment with the {ord} earliest slot",
    "pre_filter": {"enum_value": "scheduled", "phrase": "scheduled"}
                                      // or {"bool_index":0,"value":false,...}
  },

  "ambiguous": [                      // 1-2 vague-request recipes (L9)
    {"request": ["Deal with the likely no-shows.",
                 "Handle the appointments that look like no-shows."],
     "hint": "flag scheduled appointments already past their slot as no-show risks",
     "clauses": [{"enum_value": "scheduled"},
                 {"time_index": 0, "op": "LT"}],
     "action": "set_bool"}            // ONLY "delete_child" or "set_bool"
  ],

  "banks": {
    "child_msgs": ["Your appointment needs attention.",
                   "Please review your upcoming appointment.",
                   "A note about your appointment.",
                   "Your appointment may need rescheduling."],
    "agg_msgs": ["You currently top this list.",
                 "Flagging this to you directly.",
                 "You lead this count right now."],
    "fail_msgs": ["The requested operation failed.",
                  "That action could not be completed.",
                  "The last operation did not go through."]
  }
}
```

## Rules for theme authors

- **Vocabulary must be domain-native.** No kanban/CRM/project words. Field
  names, tool names, verbs, and phrases should read like the domain's real
  software would.
- **desc_style governs every tool desc:** `clean` = one crisp sentence;
  `terse` = fragment ("del appt rec"); `verbose` = two wordy sentences;
  `sloppy` = lowercase, abbreviations, maybe a typo. Styles make the eval
  honest — real tool docs are not uniform.
- **Tool naming style may vary** per theme: snake_case, camelCase, or terse
  abbreviations. Stay consistent within a theme.
- Phrases fill English templates: `{obj}` item phrase, `{name}` a
  parent's name, `{d}` day count, `{ord}` ordinal word, `{to_value}` /
  `{to_name}` from the corresponding templates.
- `placement`: `pre` = adjective before the noun ("cancelled
  appointments"), `post` = trailing phrase ("appointments already past").
- Every list needs the stated minimum entries; JSON must parse; no comments
  in the actual files.
- **`number` is the only numeric field a theme gets, and nothing sets it.**
  It exists so a request can ask for a total, an average or a largest value
  and the answer can be computed with the compute tools rather than
  guessed. Pick a quantity the domain really records per record (minutes,
  pages, units, kilometres, headcount — not money in cents), give it a
  range a person would recognise, and write `noun` as the words a request
  would use, not the field name. No tool takes it as an argument and no
  filter reads it, so a theme without one loses only the aggregate
  questions.

## Decoys and the generic tools (optional, 2026-09-23)

Two optional blocks make a theme's decoyed exam honest. Before them, decoys
came from one template bank (`harness/decoys.py` `DECOY_OPS`) in one voice
whatever the theme's `desc_style`, and the nine generic tools every world gets
(create, notes, image, writer, search) carried one template description in
every world. A classifier reading **only the description text** told real
tools from decoys at AUC 1.000 on worlds it had never seen
(`results/R10.md` §8). An exam passable that way cannot show that a model
reads the request. `harness/decoy_audit.py` is the gate: AUC ≤ 0.60 on
descriptions, on names and on teacher embeddings.

### `decoys` on every tool slot

Each of the nine `tools` slots may carry `"decoys"`: 2–4 siblings, each a
`name` and a `desc`. A decoy is a tool **with exactly the real tool's
arguments and return type** (the compiler copies them) that does something
**different** to the same record, something a request in this domain could
plausibly ask for instead. Only the description says which one the request
means. Decoys are never the right answer in normal generation. The flip probe
swaps one in as the answer (see below).

```jsonc
"delete_child": {"name": "delSeg", "desc": "del seg rec, no undo",
                 "verbs": ["delete {obj}", "wipe {obj} from the schedule"],
                 "decoys": [
                   {"name": "voidSegFare", "desc": "void all fares on seg, no undo",
                    "verbs": ["void the fares on {obj}", "kill fares for {obj}"]},
                   {"name": "purgeSegPax", "desc": "wipe pax list off seg, no undo",
                    "verbs": ["purge the pax list on {obj}", "clear pax off {obj}"]}
                 ]}
```

What each slot's decoys must fit (the signature is fixed; the action is not):

| slot | takes → returns, properties | a decoy is… |
|---|---|---|
| list_child | () → every child record, read-only | another read-only listing of the child records (e.g. "list segments with open fault reports") |
| get_child | (child id) → that child record, read-only | another read-only fetch of one child record |
| list_parent / get_parent | the same, for the parent | the same, for the parent |
| delete_child | (child id) → nothing, mutates, irreversible | another irreversible destructive action on the child |
| set_enum | (child id, a value of the enum field) → the child, mutates | another mutation that takes a value of **that same field** |
| set_bool | (child id) → the child, mutates | another one-argument change to the child |
| set_ref | (child id, parent id) → the child, mutates | another action linking the child to a parent |
| send | (parent id, message text) → nothing, external, irreversible | another outbound message or contact to the parent |

**Request templates on flip slots.** Decoys of `delete_child` and `set_bool`
carry `verbs` (≥2, each with `{obj}`); `set_ref` decoys carry `verbs` with
`{obj}` and `{to_name}` (the slot's `to_template` is reused); `send` decoys
carry `child_verbs` (with `{obj}`), `direct_verbs` (with `{name}`) and
`pair_verbs` (bare verbs), ≥2 each. They are how the flip probe writes a
request for the decoy's action.

### `v2`: the nine generic tools, in the theme's own voice

```jsonc
"v2": {
  "create":      {"name": ..., "desc": ..., "decoys": [...]},  // (parent id, [title], [enum value]) -> new child, mutates
  "list_notes":  {...},   // (child id) -> the notes on that child, read-only
  "add_note":    {...},   // (child id, text, [title]) -> the new note, mutates
  "update_note": {...},   // (child id, note id, text, [title]) -> the note, mutates
  "delete_note": {...},   // (child id, note id) -> nothing, mutates, irreversible
  "set_image":   {...},   // (child id, image URL) -> the child, mutates
  "writer":      {...},   // (brief, list of children) -> generated text, external
  "image":       {...},   // (prompt, [style]) -> an image URL, external
  "search":      {...}    // (question) -> a passage of text, read-only
}
```

All nine keys, or none. Each `desc` says what the default one says, in this
theme's `desc_style`. Each `name` is in the theme's naming style. The block
also takes `decoys` (2–4, `name` + `desc`, no request templates), subject to
the table's rule: same arguments and return, different action.

### Rules for decoy authors

- **Same granularity: change *what*, never *how much*.** The pilot
  (2026-09-23, four themes) failed the audit at description AUC 0.75 and
  name AUC 0.90 for one reason: every real tool was the plain canonical
  action (`listSubmissions`, "list all the submissions") and every decoy a
  specialisation of it (`listOverdueReviews`, "list the submissions with
  overdue reports"). An extra qualifier, an extra word in the name, a
  narrower object — each is a "decoy" feature on its own. So a decoy
  swaps the content of the real line and keeps its shape. Same
  number of words in the name. The same sentence frame, with the key noun
  phrase or verb exchanged for a sibling one of the same breadth: a parallel
  collection, source, record part or action ("list every lesson in the
  course library" / "… in the template gallery" / "… in the review
  queue"). Nothing narrower.
- **You may re-word the real line** (its `name` and `desc`, never its
  `verbs`) so the set reads as interchangeable, as long as it still says
  exactly what the tool does. Names have never been model input before spec
  0.8.0, so renaming a real tool changes no stored result. The test: shuffle
  the real line in with its decoys and give them to someone who knows the
  domain's words but not which tool the compiler wires up. If the real one
  is the plainest, shortest or most general, rewrite it or the decoys until
  nothing gives it away.
- **Same voice as the real line.** A decoy's `desc` is in the theme's
  `desc_style` and within about a third of the length of the real description
  it sits beside. Its `name` follows the real names' style (case, word
  order, abbreviations). If a classifier can tell the two apart by voice,
  length or naming habit, the exam measures style and not reading.
- **No contrast words.** Never "rather than", "instead of", "without",
  "only", "not the", "copy of", "leaving … in place". A decoy says what it
  does, as the real description does. It never says what it does not do or
  which tool it is not.
- **Domain-native.** Use the vocabulary the real descriptions use. Nothing
  generic ("reindex", "audit queue", "featured set") unless the domain
  really has it.
- **A different action.** No synonym of the real tool, and nothing another
  real tool in the theme already does. A request for the real action must not
  also fit the decoy, and the reverse.
- **Consistent with the properties.** A decoy of an irreversible slot is
  irreversible, a read-only slot's decoy only reads, and a `send` decoy
  contacts someone.
- **Names unique** across the theme: real names, the `v2` names, every
  decoy, and not `sum`/`avg`/`max`/`min`.

`python -m data.gen.domains --validate DIR` checks structure (counts,
uniqueness, request-template placeholders). `harness/decoy_audit.py` checks
style, across worlds.
