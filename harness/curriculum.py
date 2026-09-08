"""F5 curriculum: hand-written reference tasks for Levels 0-10 (plus effect-
gate tasks), in authoring form. Building them executes every reference
program through F2->F3 and derives expected states — running this module to
completion IS the Foundation exit criterion's first half.

CLI:
  python -m harness.curriculum --out data/curriculum_tasks.jsonl \
      --examples spec/examples
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from harness.authoring import resolve
from harness.context import build_context
from harness.taskbuild import build_task
from runtime.worlds import get_world

DAY = 86400
NOW = 1_760_000_000

# Each entry: id, level, world, request, constants, segments (authoring form),
# and optional error_injection / expected_status / tags.
CURRICULUM = [
    # ---------------- Level 0: direct call ----------------
    dict(
        id="L0_kanban_delete", level=0, world="kanban",
        request="Delete card 4.",
        constants=[{"type": "ID:card", "value": "card_4", "desc": "card 4"}],
        segments=["CALL @delete_card $0\nSTOP\n"],
    ),
    dict(
        id="L0_crm_close", level=0, world="crm",
        request="Close ticket 2.",
        constants=[{"type": "ID:ticket", "value": "ticket_2",
                    "desc": "ticket 2"}],
        segments=["CALL @close_ticket $0 -> r0\nSTOP\n"],
    ),
    dict(
        id="L0_projects_archive", level=0, world="projects",
        request="Archive the Website refresh project.",
        constants=[{"type": "ID:project", "value": "project_1",
                    "desc": "the Website refresh project"}],
        segments=["CALL @archive_project $0 -> r0\nSTOP\n"],
    ),
    # ---------------- Level 1: simple chain ----------------
    dict(
        id="L1_crm_email_lookup", level=1, world="crm",
        request="Look up Initech's email address and send them a payment "
                "reminder.",
        constants=[
            {"type": "ID:customer", "value": "customer_2", "desc": "Initech"},
            {"type": "STR", "value": "Your payment is past due.",
             "desc": "reminder text"},
        ],
        segments=["CALL @get_customer $0 -> r0\n"
                  "GET r0.@customer.email -> r1\n"
                  "CALL @send_email r1 $1\n"
                  "STOP\n"],
    ),
    dict(
        id="L1_kanban_msg_assignee", level=1, world="kanban",
        request="Find who card 2 is assigned to and message them that the "
                "deadline moved up.",
        constants=[
            {"type": "ID:card", "value": "card_2", "desc": "card 2"},
            {"type": "STR", "value": "The deadline for your card moved up.",
             "desc": "message text"},
        ],
        segments=["CALL @get_card $0 -> r0\n"
                  "CALL @send_message r0.@card.assignee $1\n"
                  "STOP\n"],
    ),
    dict(
        id="L1_projects_notify_owner", level=1, world="projects",
        request="Notify the owner of the Data warehouse project that it is "
                "due for review.",
        constants=[
            {"type": "ID:project", "value": "project_3",
             "desc": "the Data warehouse project"},
            {"type": "STR", "value": "Your project is due for review.",
             "desc": "notification text"},
        ],
        segments=["CALL @get_project $0 -> r0\n"
                  "CALL @notify_member r0.@project.owner $1\n"
                  "STOP\n"],
    ),
    # ---------------- Level 2: filtering ----------------
    dict(
        id="L2_kanban_archive_done", level=2, world="kanban",
        request="Archive every completed card that isn't archived yet.",
        constants=[
            {"type": "STR", "value": "done", "desc": "the completed status"},
            {"type": "BOOL", "value": False, "desc": "false"},
        ],
        segments=["CALL @list_cards -> r0\n"
                  "FILTER r0 @card.status EQ $0 AND @card.archived EQ $1 -> r1\n"
                  "FOREACH r1 -> r2\n"
                  "  CALL @archive_card r2 -> r3\n"
                  "STOP\n"],
    ),
    dict(
        id="L2_crm_close_customer_tickets", level=2, world="crm",
        request="Close all of Initech's open tickets.",
        constants=[
            {"type": "ID:customer", "value": "customer_2", "desc": "Initech"},
            {"type": "STR", "value": "open", "desc": "the open status"},
        ],
        segments=["CALL @list_tickets -> r0\n"
                  "FILTER r0 @ticket.customer EQ $0 AND @ticket.status EQ $1 -> r1\n"
                  "FOREACH r1 -> r2\n"
                  "  CALL @close_ticket r2 -> r3\n"
                  "STOP\n"],
    ),
    dict(
        id="L2_crm_charge_unpaid", level=2, world="crm",
        request="Charge Globex for their outstanding unpaid invoice.",
        constants=[
            {"type": "ID:customer", "value": "customer_3", "desc": "Globex"},
            {"type": "BOOL", "value": False, "desc": "false"},
        ],
        segments=["CALL @list_invoices -> r0\n"
                  "FILTER r0 @invoice.customer EQ $0 AND @invoice.paid EQ $1 -> r1\n"
                  "FIRST r1 -> r2\n"
                  "GET r2.@invoice.amount -> r3\n"
                  "CALL @charge_customer $0 r3\n"
                  "STOP\n"],
    ),
    # ---------------- Level 3: multiple constraints ----------------
    dict(
        id="L3_kanban_bob_overdue", level=3, world="kanban",
        request="Archive the overdue cards assigned to Bob, except the "
                "urgent ones.",
        constants=[
            {"type": "ID:user", "value": "user_1", "desc": "Bob"},
            {"type": "BOOL", "value": True, "desc": "true"},
        ],
        segments=["CALL @list_cards -> r0\n"
                  "FILTER r0 @card.assignee EQ $0 AND @card.due LT NOW "
                  "AND NOT @card.urgent EQ $1 -> r1\n"
                  "FOREACH r1 -> r2\n"
                  "  CALL @archive_card r2 -> r3\n"
                  "STOP\n"],
        tags=["adv:exception"],
    ),
    dict(
        id="L3_crm_close_low_priority", level=3, world="crm",
        request="Close every open ticket below priority 3, unless it belongs "
                "to Northwind.",
        constants=[
            {"type": "STR", "value": "open", "desc": "the open status"},
            {"type": "INT", "value": 3, "desc": "priority 3"},
            {"type": "ID:customer", "value": "customer_1",
             "desc": "Northwind"},
        ],
        segments=["CALL @list_tickets -> r0\n"
                  "FILTER r0 @ticket.status EQ $0 AND @ticket.priority LT $1 "
                  "AND NOT @ticket.customer EQ $2 -> r1\n"
                  "FOREACH r1 -> r2\n"
                  "  CALL @close_ticket r2 -> r3\n"
                  "STOP\n"],
        tags=["adv:exception", "adv:quantifier"],
    ),
    dict(
        id="L3_kanban_delete_done", level=3, world="kanban",
        request="Delete the completed cards, but keep any urgent ones.",
        constants=[
            {"type": "STR", "value": "done", "desc": "the completed status"},
            {"type": "BOOL", "value": True, "desc": "true"},
        ],
        segments=["CALL @list_cards -> r0\n"
                  "FILTER r0 @card.status EQ $0 AND NOT @card.urgent EQ $1 -> r1\n"
                  "FOREACH r1 -> r2\n"
                  "  CALL @delete_card r2\n"
                  "STOP\n"],
        tags=["adv:exception"],
    ),
    # ---------------- Level 4: aggregation ----------------
    dict(
        id="L4_crm_most_open_tickets", level=4, world="crm",
        request="Email the account manager of the customer with the most "
                "open tickets.",
        constants=[
            {"type": "STR", "value": "open", "desc": "the open status"},
            {"type": "STR",
             "value": "Your account has the most open tickets.",
             "desc": "email text"},
        ],
        segments=["CALL @list_customers -> r0\n"
                  "CALL @list_tickets -> r1\n"
                  "FILTER r1 @ticket.status EQ $0 -> r2\n"
                  "MOST r2 @ticket.customer -> r3\n"
                  "FILTER r0 @customer.id EQ r3 -> r4\n"
                  "FIRST r4 -> r5\n"
                  "CALL @get_staff r5.@customer.manager -> r6\n"
                  "GET r6.@user.email -> r7\n"
                  "CALL @send_email r7 $1\n"
                  "STOP\n"],
    ),
    dict(
        id="L4_kanban_busiest_user", level=4, world="kanban",
        request="Message whoever has the most unfinished cards on the board.",
        constants=[
            {"type": "STR", "value": "done", "desc": "the completed status"},
            {"type": "BOOL", "value": False, "desc": "false"},
            {"type": "STR", "value": "You have the most unfinished cards.",
             "desc": "message text"},
        ],
        segments=["CALL @list_cards -> r0\n"
                  "FILTER r0 NOT @card.status EQ $0 AND @card.archived EQ $1 -> r1\n"
                  "MOST r1 @card.assignee -> r2\n"
                  "CALL @send_message r2 $2\n"
                  "STOP\n"],
    ),
    # spec 0.4.0 §4: the candidate list is what makes "fewest" answerable —
    # the user with none at all is absent from the filtered list.
    dict(
        id="L4_kanban_quietest_user", level=4, world="kanban",
        request="Message whoever has the fewest unfinished cards on the "
                "board.",
        constants=[
            {"type": "STR", "value": "done", "desc": "the completed status"},
            {"type": "BOOL", "value": False, "desc": "false"},
            {"type": "STR", "value": "You have the fewest unfinished cards.",
             "desc": "message text"},
        ],
        segments=["CALL @list_users -> r0\n"
                  "CALL @list_cards -> r1\n"
                  "FILTER r1 NOT @card.status EQ $0 AND @card.archived EQ $1 -> r2\n"
                  "LEAST r2 @card.assignee r0 -> r3\n"
                  "CALL @send_message r3 $2\n"
                  "STOP\n"],
    ),
    dict(
        id="L4_crm_largest_unpaid", level=4, world="crm",
        request="Find the largest unpaid invoice and email that customer's "
                "account manager about it.",
        constants=[
            {"type": "BOOL", "value": False, "desc": "false"},
            {"type": "STR",
             "value": "Your customer holds the largest unpaid invoice.",
             "desc": "email text"},
        ],
        segments=["CALL @list_invoices -> r0\n"
                  "FILTER r0 @invoice.paid EQ $0 -> r1\n"
                  "SORT r1 @invoice.amount DESC -> r2\n"
                  "FIRST r2 -> r3\n"
                  "CALL @get_customer r3.@invoice.customer -> r4\n"
                  "CALL @get_staff r4.@customer.manager -> r5\n"
                  "GET r5.@user.email -> r6\n"
                  "CALL @send_email r6 $1\n"
                  "STOP\n"],
        tags=["adv:ordinal"],
    ),
    # ---------------- Level 5: branching ----------------
    dict(
        id="L5_crm_delinquent_branch", level=5, world="crm",
        request="Check Initech's account: if they're delinquent, alert the "
                "billing desk; otherwise send them their renewal notice.",
        constants=[
            {"type": "ID:customer", "value": "customer_2", "desc": "Initech"},
            {"type": "BOOL", "value": True, "desc": "true"},
            {"type": "STR", "value": "billing@corp.test",
             "desc": "the billing desk address"},
            {"type": "STR", "value": "Initech is delinquent.",
             "desc": "billing alert text"},
            {"type": "STR", "value": "Time to renew your plan.",
             "desc": "renewal notice text"},
        ],
        segments=["CALL @get_customer $0 -> r0\n"
                  "GET r0.@customer.delinquent -> r1\n"
                  "IF r1 EQ $1\n"
                  "  CALL @send_email $2 $3\n"
                  "ELSE\n"
                  "  GET r0.@customer.email -> r2\n"
                  "  CALL @send_email r2 $4\n"
                  "STOP\n"],
    ),
    dict(
        id="L5_kanban_done_or_nudge", level=5, world="kanban",
        request="If card 4 is finished, archive it; if not, remind its "
                "assignee to wrap it up.",
        constants=[
            {"type": "ID:card", "value": "card_4", "desc": "card 4"},
            {"type": "STR", "value": "done", "desc": "the completed status"},
            {"type": "STR", "value": "Please wrap up your card.",
             "desc": "reminder text"},
        ],
        segments=["CALL @get_card $0 -> r0\n"
                  "GET r0.@card.status -> r1\n"
                  "IF r1 EQ $1\n"
                  "  CALL @archive_card r0 -> r2\n"
                  "ELSE\n"
                  "  CALL @send_message r0.@card.assignee $2\n"
                  "STOP\n"],
    ),
    dict(
        id="L5_projects_owner_check", level=5, world="projects",
        request="If the Website refresh project's owner is no longer active, "
                "transfer it to their manager; otherwise just notify the "
                "owner it's under review.",
        constants=[
            {"type": "ID:project", "value": "project_1",
             "desc": "the Website refresh project"},
            {"type": "BOOL", "value": False, "desc": "false"},
            {"type": "STR", "value": "Your project is under review.",
             "desc": "notification text"},
        ],
        segments=["CALL @get_project $0 -> r0\n"
                  "CALL @get_member r0.@project.owner -> r1\n"
                  "GET r1.@user.active -> r2\n"
                  "IF r2 EQ $1\n"
                  "  CALL @transfer_project r0 r1.@user.manager -> r3\n"
                  "ELSE\n"
                  "  CALL @notify_member r1 $2\n"
                  "STOP\n"],
    ),
    # ---------------- Level 6: multi-stage dependency ----------------
    dict(
        id="L6_projects_transfer_inactive", level=6, world="projects",
        request="Transfer the active projects owned by inactive members to "
                "those members' managers.",
        constants=[
            {"type": "BOOL", "value": False, "desc": "false"},
            {"type": "STR", "value": "active", "desc": "the active status"},
        ],
        segments=["CALL @list_projects -> r0\n"
                  "CALL @list_members -> r1\n"
                  "FILTER r1 @user.active EQ $0 -> r2\n"
                  "FOREACH r2 -> r3\n"
                  "  FILTER r0 @project.owner EQ r3.@user.id "
                  "AND @project.status EQ $1 -> r4\n"
                  "  FOREACH r4 -> r5\n"
                  "    CALL @transfer_project r5 r3.@user.manager -> r6\n"
                  "STOP\n"],
    ),
    dict(
        id="L6_crm_alert_managers", level=6, world="crm",
        request="For every delinquent customer, email their account manager "
                "a delinquency alert.",
        constants=[
            {"type": "BOOL", "value": True, "desc": "true"},
            {"type": "STR", "value": "One of your accounts is delinquent.",
             "desc": "alert text"},
        ],
        segments=["CALL @list_customers -> r0\n"
                  "FILTER r0 @customer.delinquent EQ $0 -> r1\n"
                  "FOREACH r1 -> r2\n"
                  "  CALL @get_staff r2.@customer.manager -> r3\n"
                  "  GET r3.@user.email -> r4\n"
                  "  CALL @send_email r4 $1\n"
                  "STOP\n"],
    ),
    dict(
        id="L6_kanban_wrap_up_done", level=6, world="kanban",
        request="For each finished card still on the board, let its assignee "
                "know it's being archived, then archive it.",
        constants=[
            {"type": "STR", "value": "done", "desc": "the completed status"},
            {"type": "BOOL", "value": False, "desc": "false"},
            {"type": "STR", "value": "Your finished card is being archived.",
             "desc": "message text"},
        ],
        segments=["CALL @list_cards -> r0\n"
                  "FILTER r0 @card.status EQ $0 AND @card.archived EQ $1 -> r1\n"
                  "FOREACH r1 -> r2\n"
                  "  CALL @send_message r2.@card.assignee $2\n"
                  "  CALL @archive_card r2 -> r3\n"
                  "STOP\n"],
    ),
    # ---------------- Level 7: parallel ----------------
    dict(
        id="L7_crm_parallel_dunning", level=7, world="crm",
        request="Pull up Initech and all invoices at the same time, then "
                "re-send every unpaid Initech invoice and email them a "
                "summary.",
        constants=[
            {"type": "ID:customer", "value": "customer_2", "desc": "Initech"},
            {"type": "BOOL", "value": False, "desc": "false"},
            {"type": "STR", "value": "We re-sent your unpaid invoices.",
             "desc": "summary text"},
        ],
        segments=["PARALLEL\n"
                  "  CALL @get_customer $0 -> r0\n"
                  "  CALL @list_invoices -> r1\n"
                  "FILTER r1 @invoice.customer EQ $0 AND @invoice.paid EQ $1 -> r2\n"
                  "FOREACH r2 -> r3\n"
                  "  CALL @send_invoice r3\n"
                  "GET r0.@customer.email -> r4\n"
                  "CALL @send_email r4 $2\n"
                  "STOP\n"],
    ),
    dict(
        id="L7_kanban_parallel_check", level=7, world="kanban",
        request="Fetch the board and Bob's profile together; if Bob has any "
                "overdue cards, send him a heads-up.",
        constants=[
            {"type": "ID:user", "value": "user_1", "desc": "Bob"},
            {"type": "STR", "value": "You have overdue cards.",
             "desc": "heads-up text"},
        ],
        segments=["PARALLEL\n"
                  "  CALL @list_cards -> r0\n"
                  "  CALL @get_user $0 -> r1\n"
                  "FILTER r0 @card.assignee EQ $0 AND @card.due LT NOW -> r2\n"
                  "IF NOT EMPTY r2\n"
                  "  CALL @send_message r1 $1\n"
                  "STOP\n"],
    ),
    dict(
        id="L7_projects_parallel_transfer", level=7, world="projects",
        request="Fetch the Data warehouse project and the member list "
                "together, then hand the project to the first active member.",
        constants=[
            {"type": "ID:project", "value": "project_3",
             "desc": "the Data warehouse project"},
            {"type": "BOOL", "value": True, "desc": "true"},
        ],
        segments=["PARALLEL\n"
                  "  CALL @get_project $0 -> r0\n"
                  "  CALL @list_members -> r1\n"
                  "FILTER r1 @user.active EQ $1 -> r2\n"
                  "FIRST r2 -> r3\n"
                  "CALL @transfer_project r0 r3 -> r4\n"
                  "STOP\n"],
    ),
    # ---------------- Level 8: error recovery ----------------
    dict(
        id="L8_kanban_retry_archive", level=8, world="kanban",
        request="Archive card 1 — the board API has been flaky, so retry if "
                "it fails.",
        constants=[{"type": "ID:card", "value": "card_1", "desc": "card 1"}],
        segments=["TRY RETRY 3 -> r0\n"
                  "  CALL @archive_card $0 -> r1\n"
                  "STOP\n"],
        error_injection=[{"name": "archive_card", "code": "RATE_LIMITED",
                          "times": 2}],
    ),
    dict(
        id="L8_crm_fallback_email", level=8, world="crm",
        request="Send customer 9 their invoice reminder; if their account "
                "can't be fetched, alert the billing desk instead.",
        constants=[
            {"type": "ID:customer", "value": "customer_9",
             "desc": "customer 9"},
            {"type": "STATUS", "value": "OK", "desc": "the OK status"},
            {"type": "STR", "value": "Your invoice is due.",
             "desc": "reminder text"},
            {"type": "STR", "value": "billing@corp.test",
             "desc": "the billing desk address"},
            {"type": "STR", "value": "Could not fetch customer 9.",
             "desc": "alert text"},
        ],
        segments=["TRY -> r0\n"
                  "  CALL @get_customer $0 -> r1\n"
                  "IF r0 EQ $1\n"
                  "  GET r1.@customer.email -> r2\n"
                  "  CALL @send_email r2 $2\n"
                  "ELSE\n"
                  "  CALL @send_email $3 $4\n"
                  "STOP\n"],
        error_injection=[{"name": "get_customer", "code": "NOT_FOUND",
                          "times": -1}],
    ),
    dict(
        id="L8_projects_delete_denied", level=8, world="projects",
        request="Try to delete the Legacy CMS project; if it fails, notify "
                "Noor about the failure.",
        constants=[
            {"type": "ID:project", "value": "project_4",
             "desc": "the Legacy CMS project"},
            {"type": "STATUS", "value": "OK", "desc": "the OK status"},
            {"type": "ID:user", "value": "user_3", "desc": "Noor"},
            {"type": "STR", "value": "Deleting the Legacy CMS project failed.",
             "desc": "failure notification"},
        ],
        segments=["TRY -> r0\n"
                  "  CALL @delete_project $0\n"
                  "IF NOT r0 EQ $1\n"
                  "  CALL @notify_member $2 $3\n"
                  "STOP\n"],
        error_injection=[{"name": "delete_project",
                          "code": "PERMISSION_DENIED", "times": -1}],
    ),
    # ---------------- Level 9: ambiguous scope ----------------
    dict(
        id="L9_projects_cleanup", level=9, world="projects",
        request="Clean up the old projects.",
        constants=[
            {"type": "TIME", "value": NOW - 180 * DAY,
             "desc": "180 days ago (staleness cutoff)"},
            {"type": "STR", "value": "active", "desc": "the active status"},
        ],
        segments=["CALL @list_projects -> r0\n"
                  "FILTER r0 @project.last_activity LT $0 "
                  "AND @project.status EQ $1 -> r1\n"
                  "FOREACH r1 -> r2\n"
                  "  CALL @archive_project r2 -> r3\n"
                  "STOP\n"],
        tags=["ambiguous"],
    ),
    dict(
        id="L9_kanban_tidy", level=9, world="kanban",
        request="Tidy up the board.",
        constants=[
            {"type": "STR", "value": "done", "desc": "the completed status"},
            {"type": "BOOL", "value": False, "desc": "false"},
        ],
        segments=["CALL @list_cards -> r0\n"
                  "FILTER r0 @card.status EQ $0 AND @card.archived EQ $1 -> r1\n"
                  "FOREACH r1 -> r2\n"
                  "  CALL @archive_card r2 -> r3\n"
                  "STOP\n"],
        tags=["ambiguous"],
    ),
    dict(
        id="L9_crm_stale_tickets", level=9, world="crm",
        request="Deal with the stale open tickets.",
        constants=[
            {"type": "STR", "value": "open", "desc": "the open status"},
            {"type": "TIME", "value": NOW - 30 * DAY,
             "desc": "30 days ago (staleness cutoff)"},
        ],
        segments=["CALL @list_tickets -> r0\n"
                  "FILTER r0 @ticket.status EQ $0 AND @ticket.opened LT $1 -> r1\n"
                  "FOREACH r1 -> r2\n"
                  "  CALL @close_ticket r2 -> r3\n"
                  "STOP\n"],
        tags=["ambiguous"],
    ),
    # ---------------- Level 10: PAUSE round-trips ----------------
    dict(
        id="L10_kanban_pause_archive", level=10, world="kanban",
        request="Find the overdue unarchived cards, report back, then "
                "archive them.",
        constants=[{"type": "BOOL", "value": False, "desc": "false"}],
        segments=[
            "CALL @list_cards -> r0\n"
            "FILTER r0 @card.due LT NOW AND @card.archived EQ $0 -> r1\n"
            "PAUSE\n",
            "FOREACH r1 -> r2\n"
            "  CALL @archive_card r2 -> r3\n"
            "STOP\n",
        ],
    ),
    dict(
        id="L10_crm_pause_dunning", level=10, world="crm",
        request="Work out which invoices are overdue and unpaid, check in, "
                "then re-send each of them.",
        constants=[{"type": "BOOL", "value": False, "desc": "false"}],
        segments=[
            "CALL @list_invoices -> r0\n"
            "FILTER r0 @invoice.paid EQ $0 AND @invoice.due LT NOW -> r1\n"
            "PAUSE\n",
            "FOREACH r1 -> r2\n"
            "  CALL @send_invoice r2\n"
            "STOP\n",
        ],
    ),
    dict(
        id="L10_projects_pause_handover", level=10, world="projects",
        request="Identify the inactive members, pause for review, then move "
                "each of their active projects to their manager.",
        constants=[
            {"type": "BOOL", "value": False, "desc": "false"},
            {"type": "STR", "value": "active", "desc": "the active status"},
        ],
        segments=[
            "CALL @list_members -> r0\n"
            "FILTER r0 @user.active EQ $0 -> r1\n"
            "PAUSE\n",
            "CALL @list_projects -> r2\n"
            "FOREACH r1 -> r3\n"
            "  FILTER r2 @project.owner EQ r3.@user.id "
            "AND @project.status EQ $1 -> r4\n"
            "  FOREACH r4 -> r5\n"
            "    CALL @transfer_project r5 r3.@user.manager -> r6\n"
            "STOP\n",
        ],
    ),
    # ---------------- Effect gate (R7 groundwork) ----------------
    # ---------------- Level 11: abstain (ABORT) ----------------
    # spec §4 0.3.0: every abort that admits a referent carries one, and
    # NOT_FOUND is check-then-decline — the planner cannot know a record is
    # absent without looking, so the name is a constant and the program
    # fetches, filters, and aborts inside the IF that finds nothing.
    dict(
        id="L11_kanban_abort_not_found", level=11, world="kanban",
        request="Assign card 4 to Cyrus.",
        constants=[{"type": "ID:card", "value": "card_4", "desc": "card 4"},
                   {"type": "STR", "value": "Cyrus",
                    "desc": "the assignee named, verbatim: Cyrus"}],
        segments=[
            "CALL @list_users -> r0\n"
            "FILTER r0 @user.name EQ $1 -> r1\n"
            "IF EMPTY r1\n"
            "  ABORT NOT_FOUND $1\n"
            "FIRST r1 -> r2\n"
            "CALL @assign_card $0 r2.@user.id\n"
            "STOP\n",
        ],
        expected_status="aborted", tags=["abort"],
    ),
    dict(
        id="L11_kanban_abort_unsupported", level=11, world="kanban",
        request="Merge card 4 into card 2.",
        constants=[{"type": "ID:card", "value": "card_4", "desc": "card 4"},
                   {"type": "ID:card", "value": "card_2", "desc": "card 2"}],
        segments=["ABORT UNSUPPORTED\n"],
        expected_status="aborted", tags=["abort"],
    ),
    dict(
        id="L11_kanban_abort_needs_info", level=11, world="kanban",
        request="Create a new card for Bob.",
        constants=[{"type": "ID:user", "value": "user_1", "desc": "Bob"}],
        segments=["ABORT NEEDS_INFO @card.title\n"],
        expected_status="aborted", tags=["abort"],
    ),
    dict(
        id="L11_kanban_abort_ambiguous", level=11, world="kanban",
        request="Message him about the release.",
        constants=[{"type": "STR", "value": "The release is going out today.",
                    "desc": "message text"}],
        segments=["ABORT AMBIGUOUS @user.name\n"],
        expected_status="aborted", tags=["abort"],
    ),
    # ---------------- Level 12: FORMAT (templated text from data) ----------------
    dict(
        id="L12_kanban_duplicate_card", level=12, world="kanban",
        request="Duplicate card 4.",
        constants=[{"type": "ID:card", "value": "card_4", "desc": "card 4"},
                   {"type": "STR", "value": "Copy of {0}",
                    "desc": "title template: Copy of {0} (fill {0} with the original title)"}],
        segments=[
            "CALL @get_card $0 -> r0\n"
            "FORMAT $1 r0.@card.title -> r1\n"
            "CALL @create_card r1 r0.@card.due r0.@card.assignee -> r2\n"
            "STOP\n"],
        tags=["format"],
    ),
    dict(
        id="L12_kanban_due_reminder", level=12, world="kanban",
        request="Remind Bob that card 4 is due.",
        constants=[{"type": "ID:card", "value": "card_4", "desc": "card 4"},
                   {"type": "ID:user", "value": "user_1", "desc": "Bob"},
                   {"type": "STR", "value": "Reminder: {0} is due {1}.",
                    "desc": "message template: Reminder: {0} is due {1}. (fill {0} with the card title, {1} with its due date)"}],
        segments=[
            "CALL @get_card $0 -> r0\n"
            "FORMAT $2 r0.@card.title r0.@card.due -> r1\n"
            "CALL @send_message $1 r1\n"
            "STOP\n"],
        tags=["format"],
    ),
    # ---------------- Level 13: prose via the writer tool ----------------
    dict(
        id="L13_kanban_write_overdue", level=13, world="kanban",
        request="Tell Bob which of his cards are overdue.",
        constants=[{"type": "ID:user", "value": "user_1", "desc": "Bob"},
                   {"type": "STR", "value": "Tell Bob which of his cards are overdue.",
                    "desc": "the request itself, verbatim (brief for write_text)"}],
        segments=[
            "CALL @list_cards -> r0\n"
            "FILTER r0 @card.assignee EQ $0 AND @card.due LT NOW -> r1\n"
            "CALL @write_text $1 r1 -> r2\n"
            "CALL @send_message $0 r2\n"
            "STOP\n"],
        tags=["writer"],
    ),
    dict(
        id="GATE_kanban_delete_blocked", level=0, world="kanban",
        request="Delete card 4.",
        constants=[{"type": "ID:card", "value": "card_4", "desc": "card 4"}],
        segments=["CALL @delete_card $0\nSTOP\n"],
        expected_status="effect_blocked",
        tags=["gate"],
    ),
    dict(
        id="GATE_crm_charge_blocked", level=0, world="crm",
        request="Charge Northwind fifty dollars.",
        constants=[
            {"type": "ID:customer", "value": "customer_1",
             "desc": "Northwind"},
            {"type": "INT", "value": 5000, "desc": "amount in cents"},
        ],
        segments=["CALL @charge_customer $0 $1\nSTOP\n"],
        expected_status="effect_blocked",
        tags=["gate"],
    ),
]


def build_curriculum(seed_base: int = 1000) -> list:
    tasks = []
    for i, spec in enumerate(CURRICULUM):
        seed = seed_base + i * 101
        world = get_world(spec["world"])
        ctx, sandbox_ctx = build_context(
            world, spec["constants"], random.Random(seed))
        segments = [resolve(s, ctx) for s in spec["segments"]]
        task = build_task(
            task_id=spec["id"], level=spec["level"],
            world_name=spec["world"], request=spec["request"],
            constants=spec["constants"], segments=segments, seed=seed,
            error_injection=spec.get("error_injection"),
            expected_status=spec.get("expected_status", "ok"),
            tags=spec.get("tags"),
            provenance={"source": "curriculum", "seed": seed,
                        "authoring": spec["segments"]},
            prebuilt=(ctx, sandbox_ctx))
        tasks.append(task)
    return tasks


def write_examples(out_dir: Path):
    """spec/examples/: the hand-written programs in authoring form."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in CURRICULUM:
        lines = [
            f"# {spec['id']}  (level {spec['level']}, world {spec['world']})",
            f"# request: {spec['request']}",
        ]
        for i, c in enumerate(spec["constants"]):
            lines.append(f"# ${i}: {c['type']} = {c['value']!r} — {c['desc']}")
        if spec.get("error_injection"):
            lines.append(f"# injected errors: {spec['error_injection']}")
        if spec.get("expected_status"):
            lines.append(f"# expected status: {spec['expected_status']}")
        lines.append("")
        for i, seg in enumerate(spec["segments"]):
            if i > 0:
                lines.append("# --- continuation segment (after PAUSE) ---")
            lines.append(seg.rstrip("\n"))
        (out_dir / f"{spec['id']}.ac").write_text("\n".join(lines) + "\n",
                                                 encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/curriculum_tasks.jsonl")
    ap.add_argument("--examples", default=None)
    args = ap.parse_args()
    tasks = build_curriculum()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(f"wrote {len(tasks)} tasks -> {out}")
    if args.examples:
        write_examples(Path(args.examples))
        print(f"wrote examples -> {args.examples}")


if __name__ == "__main__":
    main()
