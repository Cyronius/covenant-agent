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
    ]
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
