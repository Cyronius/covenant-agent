"""Customer service under a written policy: retail, airline and telecom.

Borrowed from tau2-bench (github.com/sierra-research/tau2-bench, MIT,
Sierra Research; see data/borrowed/tau2/README.md). The entities are a
practical subset of tau2's domain databases, the tools are named and
described after tau2's tools, and the rules tau2's written policy states are
enforced by runtime/engines/service.js: a request the policy forbids fails
with PERMISSION_DENIED and a message that names the rule
("policy: only a pending order can be cancelled (order #W1 is delivered)").

Three worlds share this module:

  service_retail    orders, items, product variants, payment methods
  service_airline   reservations, flight segments, dated flights, passengers
  service_telecom   lines, plans, bills and the customer's phone (held out:
                    the exam of the three, .claude/plans/borrowed-worlds.md
                    "Keeping the exams honest")

Where tau2 passes a list (return these item ids, update these passengers) a
tool here takes one record and is called once per record: the language has
no list literals, and a planner that names each item is what the brief
asks for anyway. tau2's "once per order" rule survives as the order's
`locked` flag: an order whose items were already modified, returned or
exchanged in an earlier conversation refuses a second request.

Money is INT cents everywhere. tau2's transfer_to_human_agents has no tool
here: a request that cannot be handled within policy is `ABORT UNSUPPORTED`,
the program's own way of declining.
"""
from __future__ import annotations

import calendar
import time as _time


def _epoch(stamp: str) -> int:
    """'2024-05-15T15:00:00' read as UTC (the policy's EST clock, shifted
    uniformly; only differences are ever compared)."""
    return calendar.timegm(_time.strptime(stamp, "%Y-%m-%dT%H:%M:%S"))


RETAIL_NOW = _epoch("2024-05-15T15:00:00")
AIRLINE_NOW = _epoch("2024-05-15T15:00:00")      # tau2 airline policy's clock
TELECOM_NOW = _epoch("2025-02-25T12:08:00")      # tau2 telecom policy's clock
AIRLINE_TODAY = "2024-05-15"
TELECOM_TODAY = "2025-02-25"


def _engine(fn: str) -> dict:
    return {"op": "engine", "module": "service", "fn": fn}


def _p(name, type_, desc, field=None, required=True) -> dict:
    p = {"name": name, "type": type_, "desc": desc}
    if field:
        p["field"] = list(field)
    if not required:
        p["required"] = False
    return p


def _t(name, desc, params, returns, effects, impl) -> dict:
    return {"name": name, "desc": desc, "params": params, "returns": returns,
            "effects": effects, "impl": impl}


# ================================================================== retail

RETAIL_ORDER_STATUS = ["pending", "processed", "delivered", "cancelled",
                       "pending (item modified)", "return requested",
                       "exchange requested"]
RETAIL_CANCEL_REASONS = ["no longer needed", "ordered by mistake"]

_ADDR = ("address1", "address2", "city", "state", "zip")

RETAIL = {
    "name": "service_retail",
    "now": RETAIL_NOW,
    "policy": "tau2-bench retail policy.md (adapted)",
    "entities": {
        "user": {"id": "ID:user", "first_name": "STR", "last_name": "STR",
                 "email": "STR", "address1": "STR", "address2": "STR",
                 "city": "STR", "state": "STR", "zip": "STR"},
        "payment_method": {"id": "ID:payment_method", "user": "ID:user",
                           "kind": "STR", "balance": "INT",
                           "last_four": "STR"},
        "product": {"id": "ID:product", "name": "STR"},
        "variant": {"id": "ID:variant", "product": "ID:product",
                    "options": "STR", "available": "BOOL", "price": "INT"},
        "order": {"id": "ID:order", "user": "ID:user", "status": "STR",
                  "address1": "STR", "address2": "STR", "city": "STR",
                  "state": "STR", "zip": "STR",
                  "payment_method": "ID:payment_method", "total": "INT",
                  "cancel_reason": "STR", "locked": "BOOL"},
        "order_item": {"id": "ID:order_item", "order": "ID:order",
                       "product": "ID:product", "variant": "ID:variant",
                       "name": "STR", "price": "INT", "status": "STR",
                       "new_variant": "ID:variant",
                       "payment_method": "ID:payment_method"},
    },
    "enums": {("order", "status"): RETAIL_ORDER_STATUS,
              ("order", "cancel_reason"): RETAIL_CANCEL_REASONS,
              ("payment_method", "kind"): ["gift_card", "credit_card",
                                           "paypal"]},
    "tools": [
        _t("find_user_id_by_email",
           "Find a customer's user id by their email address. Every "
           "conversation starts by authenticating the customer this way or "
           "by name and zip code.",
           [_p("email", "STR", "the customer's email", ("user", "email"))],
           "ID:user", [], _engine("retail_find_user_id_by_email")),
        _t("find_user_id_by_name_zip",
           "Find a customer's user id by first name, last name and zip code, "
           "when they cannot give their email.",
           [_p("first_name", "STR", "first name", ("user", "first_name")),
            _p("last_name", "STR", "last name", ("user", "last_name")),
            _p("zip", "STR", "zip code", ("user", "zip"))],
           "ID:user", [], _engine("retail_find_user_id_by_name_zip")),
        _t("get_user_details",
           "Get a customer's profile: name, email and default address.",
           [_p("user", "ID:user", "the customer", ("user", "id"))],
           "OBJ:user", [], {"op": "get", "entity": "user", "id_param": 0}),
        _t("list_user_orders", "List every order a customer has placed.",
           [_p("user", "ID:user", "the customer", ("user", "id"))],
           "LIST OBJ:order", [],
           {"op": "list_by", "entity": "order", "field": "user",
            "id_param": 0}),
        _t("list_user_payment_methods",
           "List the payment methods saved in a customer's profile: gift "
           "cards (with balance), credit cards and PayPal.",
           [_p("user", "ID:user", "the customer", ("user", "id"))],
           "LIST OBJ:payment_method", [],
           {"op": "list_by", "entity": "payment_method", "field": "user",
            "id_param": 0}),
        _t("get_order_details",
           "Get an order's status, shipping address, total and the payment "
           "method it was paid with.",
           [_p("order", "ID:order", "the order, such as '#W0000000'",
               ("order", "id"))],
           "OBJ:order", [], {"op": "get", "entity": "order", "id_param": 0}),
        _t("list_order_items", "List the items in an order.",
           [_p("order", "ID:order", "the order", ("order", "id"))],
           "LIST OBJ:order_item", [],
           {"op": "list_by", "entity": "order_item", "field": "order",
            "id_param": 0}),
        _t("get_product_details", "Get a product type's name.",
           [_p("product", "ID:product", "the product type",
               ("product", "id"))],
           "OBJ:product", [], {"op": "get", "entity": "product",
                               "id_param": 0}),
        _t("list_product_variants",
           "List every variant of a product type with its options, price "
           "and availability.",
           [_p("product", "ID:product", "the product type",
               ("product", "id"))],
           "LIST OBJ:variant", [],
           {"op": "list_by", "entity": "variant", "field": "product",
            "id_param": 0}),
        _t("get_item_details",
           "Get one product variant's options, price and availability.",
           [_p("variant", "ID:variant", "the variant (item id)",
               ("variant", "id"))],
           "OBJ:variant", [], {"op": "get", "entity": "variant",
                               "id_param": 0}),
        _t("list_all_product_types",
           "List the store's product types (there are about 50).",
           [], "LIST OBJ:product", [], {"op": "list", "entity": "product"}),
        _t("cancel_pending_order",
           "Cancel a pending order and refund it. Only a pending order can be "
           "cancelled, and the reason must be 'no longer needed' or 'ordered "
           "by mistake'; other reasons are not acceptable.",
           [_p("order", "ID:order", "the order to cancel", ("order", "id")),
            _p("reason", "STR", "the cancellation reason",
               ("order", "cancel_reason"))],
           "OBJ:order", ["mutates", "irreversible"],
           _engine("retail_cancel_pending_order")),
        _t("modify_pending_order_address",
           "Change the shipping address of a pending order.",
           [_p("order", "ID:order", "the pending order", ("order", "id"))]
           + [_p(f, "STR", f"new {f}", ("order", f)) for f in _ADDR],
           "OBJ:order", ["mutates"],
           _engine("retail_modify_pending_order_address")),
        _t("modify_pending_order_payment",
           "Pay for a pending order with a different payment method from the "
           "customer's profile. A gift card must cover the whole total.",
           [_p("order", "ID:order", "the pending order", ("order", "id")),
            _p("payment_method", "ID:payment_method", "the new payment method",
               ("payment_method", "id"))],
           "OBJ:order", ["mutates"],
           _engine("retail_modify_pending_order_payment")),
        _t("modify_pending_order_items",
           "Change an item of a pending order to another available option of "
           "the same product (call once per item); the payment method pays "
           "or receives the price difference. Items of an order can only be "
           "modified in one request.",
           [_p("item", "ID:order_item", "the item in the order",
               ("order_item", "id")),
            _p("new_item", "ID:variant", "the variant to change it to",
               ("variant", "id")),
            _p("payment_method", "ID:payment_method",
               "pays or receives the price difference",
               ("payment_method", "id"))],
           "OBJ:order", ["mutates"],
           _engine("retail_modify_pending_order_items")),
        _t("exchange_delivered_order_items",
           "Exchange an item of a delivered order for another available "
           "option of the same product (call once per item); the payment "
           "method pays or receives the price difference. A delivered order "
           "can be returned or exchanged only once.",
           [_p("item", "ID:order_item", "the item in the order",
               ("order_item", "id")),
            _p("new_item", "ID:variant", "the variant to exchange it for",
               ("variant", "id")),
            _p("payment_method", "ID:payment_method",
               "pays or receives the price difference",
               ("payment_method", "id"))],
           "OBJ:order", ["mutates"],
           _engine("retail_exchange_delivered_order_items")),
        _t("return_delivered_order_items",
           "Return an item of a delivered order (call once per item). The "
           "refund must go to the order's original payment method or to a "
           "gift card. A delivered order can be returned or exchanged only "
           "once.",
           [_p("item", "ID:order_item", "the item to return",
               ("order_item", "id")),
            _p("payment_method", "ID:payment_method", "receives the refund",
               ("payment_method", "id"))],
           "OBJ:order", ["mutates"],
           _engine("retail_return_delivered_order_items")),
        _t("modify_user_address",
           "Change a customer's default address.",
           [_p("user", "ID:user", "the customer", ("user", "id"))]
           + [_p(f, "STR", f"new {f}", ("user", f)) for f in _ADDR],
           "OBJ:user", ["mutates"], _engine("retail_modify_user_address")),
    ],
    "default_state": {"entities": {"user": [], "payment_method": [],
                                   "product": [], "variant": [], "order": [],
                                   "order_item": []},
                      "outbox": [], "payments": []},
}


# ================================================================= airline

AIRLINE_CABINS = ["basic_economy", "economy", "business"]
AIRLINE_CANCEL_REASONS = ["change of plan", "airline cancelled flight",
                          "health", "weather", "other"]

AIRLINE = {
    "name": "service_airline",
    "now": AIRLINE_NOW,
    "policy": "tau2-bench airline policy.md (adapted)",
    "entities": {
        "user": {"id": "ID:user", "first_name": "STR", "last_name": "STR",
                 "email": "STR", "membership": "STR"},
        "payment_method": {"id": "ID:payment_method", "user": "ID:user",
                           "kind": "STR", "balance": "INT",
                           "last_four": "STR"},
        "reservation": {"id": "ID:reservation", "user": "ID:user",
                        "origin": "STR", "destination": "STR",
                        "trip_type": "STR", "cabin": "STR",
                        "passenger_count": "INT", "total_baggages": "INT",
                        "nonfree_baggages": "INT", "insurance": "BOOL",
                        "created": "TIME", "status": "STR",
                        "payment_method": "ID:payment_method",
                        "cancel_reason": "STR", "changed": "BOOL"},
        "segment": {"id": "ID:segment", "reservation": "ID:reservation",
                    "flight": "ID:flight", "origin": "STR",
                    "destination": "STR", "date": "STR", "price": "INT"},
        "flight": {"id": "ID:flight", "number": "STR", "origin": "STR",
                   "destination": "STR", "date": "STR", "status": "STR",
                   "departs": "STR", "price_basic_economy": "INT",
                   "price_economy": "INT", "price_business": "INT"},
        "passenger": {"id": "ID:passenger", "reservation": "ID:reservation",
                      "first_name": "STR", "last_name": "STR", "dob": "STR"},
    },
    "enums": {("reservation", "cabin"): AIRLINE_CABINS,
              ("reservation", "cancel_reason"): AIRLINE_CANCEL_REASONS,
              ("user", "membership"): ["regular", "silver", "gold"]},
    "tools": [
        _t("get_user_details",
           "Get a customer's profile, including membership level.",
           [_p("user", "ID:user", "the customer", ("user", "id"))],
           "OBJ:user", [], {"op": "get", "entity": "user", "id_param": 0}),
        _t("list_user_reservations",
           "List every reservation a customer holds.",
           [_p("user", "ID:user", "the customer", ("user", "id"))],
           "LIST OBJ:reservation", [],
           {"op": "list_by", "entity": "reservation", "field": "user",
            "id_param": 0}),
        _t("list_user_payment_methods",
           "List the payment methods in a customer's profile: credit cards, "
           "gift cards and travel certificates.",
           [_p("user", "ID:user", "the customer", ("user", "id"))],
           "LIST OBJ:payment_method", [],
           {"op": "list_by", "entity": "payment_method", "field": "user",
            "id_param": 0}),
        _t("get_reservation_details",
           "Get a reservation's trip, cabin, passengers, bags, insurance and "
           "booking time.",
           [_p("reservation", "ID:reservation", "the reservation, such as "
               "'ZFA04Y'", ("reservation", "id"))],
           "OBJ:reservation", [],
           {"op": "get", "entity": "reservation", "id_param": 0}),
        _t("list_reservation_flights",
           "List the flight segments of a reservation.",
           [_p("reservation", "ID:reservation", "the reservation",
               ("reservation", "id"))],
           "LIST OBJ:segment", [],
           {"op": "list_by", "entity": "segment", "field": "reservation",
            "id_param": 0}),
        _t("list_reservation_passengers",
           "List the passengers on a reservation.",
           [_p("reservation", "ID:reservation", "the reservation",
               ("reservation", "id"))],
           "LIST OBJ:passenger", [],
           {"op": "list_by", "entity": "passenger", "field": "reservation",
            "id_param": 0}),
        _t("get_flight_status",
           "Get a dated flight's status (available, on time, delayed, "
           "flying, landed, cancelled) and its prices per cabin.",
           [_p("flight", "ID:flight", "the flight on a date",
               ("flight", "id"))],
           "OBJ:flight", [], {"op": "get", "entity": "flight",
                              "id_param": 0}),
        _t("search_direct_flight",
           "Search direct flights between two airports on a date.",
           [_p("origin", "STR", "origin airport code", ("flight", "origin")),
            _p("destination", "STR", "destination airport code",
               ("flight", "destination")),
            _p("date", "STR", "date, such as '2024-05-20'",
               ("flight", "date"))],
           "LIST OBJ:flight", [], _engine("airline_search_direct_flight")),
        _t("cancel_reservation",
           "Cancel a whole reservation and refund it to the original payment. "
           "Not possible once any flight has been flown. Otherwise allowed "
           "only if it was booked within the last 24 hours, a flight was "
           "cancelled by the airline, the cabin is business, or it has "
           "travel insurance and the reason is health or weather.",
           [_p("reservation", "ID:reservation", "the reservation to cancel",
               ("reservation", "id")),
            _p("reason", "STR", "the reason for cancellation",
               ("reservation", "cancel_reason"))],
           "OBJ:reservation", ["mutates", "irreversible"],
           _engine("airline_cancel_reservation")),
        _t("update_reservation_cabin",
           "Change the cabin class of every flight in a reservation, paying "
           "or refunding the difference. Not possible once any flight has "
           "been flown; the cabin is always the same across a reservation. "
           "Pay with a gift card or credit card, never a certificate.",
           [_p("reservation", "ID:reservation", "the reservation",
               ("reservation", "id")),
            _p("cabin", "STR", "the new cabin class", ("reservation",
                                                        "cabin")),
            _p("payment_method", "ID:payment_method",
               "pays or receives the difference", ("payment_method", "id"))],
           "OBJ:reservation", ["mutates"],
           _engine("airline_update_reservation_cabin")),
        _t("update_reservation_flights",
           "Move one flight segment of a reservation to another flight with "
           "the same origin and destination, paying or refunding the "
           "difference. Basic economy flights cannot be modified. Pay with a "
           "gift card or credit card.",
           [_p("segment", "ID:segment", "the segment to change",
               ("segment", "id")),
            _p("flight", "ID:flight", "the new flight on its date",
               ("flight", "id")),
            _p("payment_method", "ID:payment_method",
               "pays or receives the difference", ("payment_method", "id"))],
           "OBJ:reservation", ["mutates"],
           _engine("airline_update_reservation_flights")),
        _t("update_reservation_baggages",
           "Set the total number of checked bags on a reservation. Bags can "
           "be added but not removed; bags beyond the free allowance "
           "(by membership and cabin) cost $50 each.",
           [_p("reservation", "ID:reservation", "the reservation",
               ("reservation", "id")),
            _p("total_baggages", "INT", "the new total of checked bags",
               ("reservation", "total_baggages")),
            _p("payment_method", "ID:payment_method", "pays for extra bags",
               ("payment_method", "id"))],
           "OBJ:reservation", ["mutates"],
           _engine("airline_update_reservation_baggages")),
        _t("update_reservation_passengers",
           "Change one passenger's details on a reservation. The number of "
           "passengers can never change.",
           [_p("passenger", "ID:passenger", "the passenger to change",
               ("passenger", "id")),
            _p("first_name", "STR", "first name", ("passenger",
                                                  "first_name")),
            _p("last_name", "STR", "last name", ("passenger", "last_name")),
            _p("dob", "STR", "date of birth, such as '1990-04-05'",
               ("passenger", "dob"))],
           "OBJ:reservation", ["mutates"],
           _engine("airline_update_reservation_passengers")),
        _t("send_certificate",
           "Send a travel certificate as compensation, only when asked for, "
           "and only to silver or gold members, travellers with insurance, "
           "or business class: $100 per passenger for a cancelled flight, "
           "or $50 per passenger for a delayed flight once the reservation "
           "has been changed or cancelled.",
           [_p("reservation", "ID:reservation", "the affected reservation",
               ("reservation", "id")),
            _p("amount", "INT", "the certificate amount in cents",
               ("payment_method", "balance"))],
           "OBJ:payment_method", ["mutates"],
           _engine("airline_send_certificate")),
    ],
    "default_state": {"entities": {"user": [], "payment_method": [],
                                   "reservation": [], "segment": [],
                                   "flight": [], "passenger": []},
                      "outbox": [], "payments": []},
}


# ================================================================= telecom

TELECOM_NETWORK_MODES = ["4g_5g_preferred", "4g_only", "3g_only", "2g_only"]


def _phone_tool(name: str, desc: str) -> dict:
    return _t(name, desc,
              [_p("phone", "ID:phone", "the customer's phone",
                  ("phone", "id"))],
              "OBJ:phone", ["mutates"], _engine(f"telecom_{name}"))


TELECOM = {
    "name": "service_telecom",
    "now": TELECOM_NOW,
    "policy": "tau2-bench telecom main_policy.md and tech_support_manual.md "
              "(adapted)",
    "entities": {
        "customer": {"id": "ID:customer", "name": "STR",
                     "phone_number": "STR", "dob": "STR", "status": "STR"},
        "plan": {"id": "ID:plan", "name": "STR", "data_limit_gb": "INT",
                 "refuel_price_per_gb": "INT", "monthly_price": "INT"},
        "line": {"id": "ID:line", "customer": "ID:customer",
                 "phone_number": "STR", "status": "STR", "plan": "ID:plan",
                 "data_used_mb": "INT", "data_refuel_gb": "INT",
                 "roaming_enabled": "BOOL", "contract_end": "STR"},
        "bill": {"id": "ID:bill", "customer": "ID:customer",
                 "amount": "INT", "status": "STR", "due": "STR"},
        "phone": {"id": "ID:phone", "line": "ID:line",
                  "airplane_mode": "BOOL", "network_mode": "STR",
                  "wifi_calling": "BOOL", "apn": "STR", "mobile_data": "BOOL",
                  "data_saver": "BOOL", "roaming": "BOOL", "vpn": "BOOL",
                  "sim": "STR", "abroad": "BOOL", "sms_permission": "BOOL",
                  "storage_permission": "BOOL", "needs_reboot": "BOOL"},
    },
    "enums": {("phone", "network_mode"): TELECOM_NETWORK_MODES,
              ("line", "status"): ["Active", "Suspended"],
              ("bill", "status"): ["Paid", "Overdue", "Awaiting Payment"]},
    "tools": [
        _t("get_customer_by_phone",
           "Find a customer by their phone number. Identify the customer "
           "before anything else.",
           [_p("phone_number", "STR", "a phone number such as "
               "'555-123-2002'", ("customer", "phone_number"))],
           "OBJ:customer", [], _engine("telecom_get_customer_by_phone")),
        _t("get_customer_by_id", "Get a customer's profile by id.",
           [_p("customer", "ID:customer", "the customer",
               ("customer", "id"))],
           "OBJ:customer", [], {"op": "get", "entity": "customer",
                                "id_param": 0}),
        _t("list_customer_lines", "List a customer's phone lines.",
           [_p("customer", "ID:customer", "the customer",
               ("customer", "id"))],
           "LIST OBJ:line", [],
           {"op": "list_by", "entity": "line", "field": "customer",
            "id_param": 0}),
        _t("get_line_details",
           "Get a line's status, plan, data used this month, refuelled data, "
           "roaming and contract end date.",
           [_p("line", "ID:line", "the line", ("line", "id"))],
           "OBJ:line", [], {"op": "get", "entity": "line", "id_param": 0}),
        _t("get_plan_details",
           "Get a plan's data limit, monthly price and data refuelling "
           "price per GB.",
           [_p("plan", "ID:plan", "the plan", ("plan", "id"))],
           "OBJ:plan", [], {"op": "get", "entity": "plan", "id_param": 0}),
        _t("get_bills_for_customer", "List a customer's bills.",
           [_p("customer", "ID:customer", "the customer",
               ("customer", "id"))],
           "LIST OBJ:bill", [],
           {"op": "list_by", "entity": "bill", "field": "customer",
            "id_param": 0}),
        _t("send_payment_request",
           "Send the customer a payment request for an overdue bill; the "
           "bill becomes Awaiting Payment. Only an overdue bill, and only "
           "one awaiting payment at a time.",
           [_p("bill", "ID:bill", "the overdue bill", ("bill", "id"))],
           "OBJ:bill", ["mutates"], _engine("telecom_send_payment_request")),
        _t("make_payment",
           "Have the customer accept the payment request on a bill that is "
           "awaiting payment; the bill becomes Paid.",
           [_p("bill", "ID:bill", "the bill awaiting payment",
               ("bill", "id"))],
           "OBJ:bill", ["mutates"], _engine("telecom_make_payment")),
        _t("resume_line",
           "Lift a line's suspension once every overdue bill is paid. Not "
           "allowed when the line's contract end date is in the past. The "
           "phone must be rebooted afterwards.",
           [_p("line", "ID:line", "the suspended line", ("line", "id"))],
           "OBJ:line", ["mutates"], _engine("telecom_resume_line")),
        _t("enable_roaming",
           "Enable data roaming on a line, free of charge, for a customer "
           "travelling outside the home network.",
           [_p("line", "ID:line", "the line", ("line", "id"))],
           "OBJ:line", ["mutates"], _engine("telecom_enable_roaming")),
        _t("disable_roaming", "Disable data roaming on a line.",
           [_p("line", "ID:line", "the line", ("line", "id"))],
           "OBJ:line", ["mutates"], _engine("telecom_disable_roaming")),
        _t("refuel_data",
           "Add data to a line whose usage exceeded its plan limit, at the "
           "plan's price per GB. At most 2 GB can be refuelled.",
           [_p("line", "ID:line", "the line", ("line", "id")),
            _p("gb_amount", "INT", "GB of data to add",
               ("line", "data_refuel_gb"))],
           "OBJ:line", ["mutates"], _engine("telecom_refuel_data")),
        _phone_tool("toggle_airplane_mode",
                    "Walk the customer through switching airplane mode on "
                    "or off; with it on there is no cellular connection."),
        _t("set_network_mode_preference",
           "Walk the customer through setting the phone's preferred network "
           "mode (4g_5g_preferred, 4g_only, 3g_only or 2g_only).",
           [_p("phone", "ID:phone", "the customer's phone", ("phone", "id")),
            _p("mode", "STR", "the network mode", ("phone",
                                                  "network_mode"))],
           "OBJ:phone", ["mutates"],
           _engine("telecom_set_network_mode_preference")),
        _phone_tool("toggle_wifi_calling",
                    "Walk the customer through switching Wi-Fi calling on "
                    "or off; with MMS over Wi-Fi it can stop MMS sending."),
        _phone_tool("reset_apn_settings",
                    "Walk the customer through resetting the APN settings "
                    "to default; takes effect after a reboot."),
        _phone_tool("reboot_device",
                    "Walk the customer through restarting the phone."),
        _t("grant_app_permission",
           "Walk the customer through granting an app a permission, such as "
           "the messaging app's sms or storage permission.",
           [_p("phone", "ID:phone", "the customer's phone", ("phone", "id")),
            _p("app_name", "STR", "the app, such as 'messaging'"),
            _p("permission", "STR", "the permission, such as 'sms'")],
           "OBJ:phone", ["mutates"], _engine("telecom_grant_app_permission")),
        _phone_tool("toggle_data",
                    "Walk the customer through switching mobile data on or "
                    "off."),
        _phone_tool("reseat_sim_card",
                    "Walk the customer through taking the SIM card out and "
                    "putting it back in."),
        _phone_tool("toggle_roaming",
                    "Walk the customer through switching data roaming on "
                    "or off on the phone itself."),
        _phone_tool("disconnect_vpn",
                    "Walk the customer through disconnecting a VPN."),
        _phone_tool("toggle_data_saver_mode",
                    "Walk the customer through switching data saver mode on "
                    "or off; with it on data is slowed down."),
    ],
    "default_state": {"entities": {"customer": [], "plan": [], "line": [],
                                   "bill": [], "phone": []},
                      "outbox": [], "payments": []},
}

# Description-only siblings (harness/decoys.py `authored_decoys`): same
# signature, a neighbouring job, a noop impl when drawn. Written in the
# tools' own style so the description, not the style, has to decide. Only a
# generator run with --decoys draws them; the registered tools are unchanged.
AUTHORED_DECOYS = {
    "cancel_pending_order": [
        ("hold_pending_order", "Put a pending order on hold so it does not "
         "ship until the customer releases it; the reason is noted on the "
         "order."),
        ("flag_pending_order", "Flag a pending order for a fraud review "
         "before it ships; the reason goes to the review team.")],
    "modify_pending_order_address": [
        ("add_order_delivery_instructions", "Attach delivery instructions "
         "(a gate or building address) to a pending order for the courier; "
         "the shipping address itself is unchanged."),
        ("set_order_billing_address", "Change the billing address recorded "
         "on a pending order; it still ships to its shipping address.")],
    "modify_pending_order_payment": [
        ("split_pending_order_payment", "Add a second payment method to a "
         "pending order so its total is split between the two."),
        ("refund_pending_order_payment", "Refund a pending order's total to "
         "a payment method without cancelling the order.")],
    "modify_pending_order_items": [
        ("add_pending_order_item", "Add one more unit of a variant next to "
         "an item of a pending order, charged to the payment method."),
        ("price_match_order_item", "Price-match an item of an order to a "
         "cheaper variant and refund the difference to the payment method; "
         "the item itself is unchanged.")],
    "exchange_delivered_order_items": [
        ("reorder_delivered_order_item", "Place a new order for another "
         "option of a delivered item, charged to the payment method; the "
         "delivered item is kept.")],
    "return_delivered_order_items": [
        ("report_damaged_order_item", "Report an item of a delivered order "
         "as damaged in transit; the payment method gets a goodwill credit "
         "once the carrier confirms."),
        ("refund_order_item_keep_item", "Refund an item of a delivered "
         "order to a payment method while the customer keeps the item.")],
    "modify_user_address": [
        ("add_user_address", "Save an extra address in a customer's address "
         "book; the default address stays as it is."),
        ("set_user_billing_address", "Change the billing address on a "
         "customer's profile; the default shipping address is unchanged.")],
    "get_order_details": [
        ("get_order_tracking", "Get the carrier tracking details of an "
         "order's shipment.")],
    "cancel_reservation": [
        ("convert_reservation_to_credit", "Turn a reservation into airline "
         "credit for a future trip, keeping the booking reference; the reason "
         "is logged."),
        ("waitlist_reservation_refund", "Put a reservation on the refund "
         "waitlist for a supervisor to decide; the reason is logged.")],
    "update_reservation_cabin": [
        ("request_upgrade_waitlist", "Put every passenger of a reservation "
         "on the upgrade waitlist for a cabin; the payment method is charged "
         "only if the upgrade clears.")],
    "update_reservation_flights": [
        ("add_standby_flight", "Put a segment's passengers on standby for "
         "another flight on the same route; the payment method is charged "
         "if seats are confirmed.")],
    "update_reservation_baggages": [
        ("prepay_carry_on_bags", "Prepay a number of carry-on bags for a "
         "reservation with the payment method.")],
    "update_reservation_passengers": [
        ("add_known_traveler_info", "Record a passenger's name and date of "
         "birth for airport security pre-check, without changing the "
         "ticket.")],
    "send_certificate": [
        ("send_meal_voucher", "Send an airport meal voucher of an amount "
         "for a reservation's passengers while they wait.")],
}
for _w in (RETAIL, AIRLINE):
    for _tool in _w["tools"]:
        if _tool["name"] in AUTHORED_DECOYS:
            _tool["authored_decoys"] = [
                {"name": n, "desc": d} for n, d in
                AUTHORED_DECOYS[_tool["name"]]]

WORLDS = [RETAIL, AIRLINE, TELECOM]
TRAINABLE = ["service_retail", "service_airline"]
HELD_OUT = ["service_telecom"]


def register() -> None:
    """Add the three worlds to the registry at runtime. runtime/worlds/
    __init__.py does not import this module yet (shared file; see the
    report that added it), so generators and tests call this."""
    from runtime import worlds as W
    for w in WORLDS:
        W.WORLDS[w["name"]] = w
