"""Projects world: projects, owners, managers. HELD OUT for R5 —
reserved in data/holdout/reserved.json and never enters training data."""

DAY = 86400
NOW = 1_760_000_000

WORLD = {
    "name": "projects",
    "now": NOW,
    "holdout": True,
    "entities": {
        "project": {
            "id": "ID:project",
            "name": "STR",
            "owner": "ID:user",
            "status": "STR",          # active | archived
            "last_activity": "TIME",
        },
        "user": {
            "id": "ID:user",
            "name": "STR",
            "email": "STR",
            "manager": "ID:user",
            "active": "BOOL",
        },
    },
    "enums": {("project", "status"): ["active", "archived"]},
    "tools": [
        {
            "name": "list_projects",
            "desc": "List all projects in the workspace.",
            "params": [],
            "returns": "LIST OBJ:project",
            "effects": ["READ"],
            "impl": {"op": "list", "entity": "project"},
        },
        {
            "name": "get_project",
            "desc": "Fetch one project by id.",
            "params": [{"name": "project", "type": "ID:project",
                        "desc": "the project", "field": ["project", "id"]}],
            "returns": "OBJ:project",
            "effects": ["READ"],
            "impl": {"op": "get", "entity": "project", "id_param": 0},
        },
        {
            "name": "list_members",
            "desc": "List all workspace members.",
            "params": [],
            "returns": "LIST OBJ:user",
            "effects": ["READ"],
            "impl": {"op": "list", "entity": "user"},
        },
        {
            "name": "get_member",
            "desc": "Fetch one workspace member by id.",
            "params": [{"name": "user", "type": "ID:user", "desc": "the member",
                        "field": ["user", "id"]}],
            "returns": "OBJ:user",
            "effects": ["READ"],
            "impl": {"op": "get", "entity": "user", "id_param": 0},
        },
        {
            "name": "transfer_project",
            "desc": "Transfer a project to a new owner.",
            "params": [
                {"name": "project", "type": "ID:project", "desc": "the project",
                 "field": ["project", "id"]},
                {"name": "owner", "type": "ID:user", "desc": "the new owner",
                 "field": ["project", "owner"]},
            ],
            "returns": "OBJ:project",
            "effects": ["WRITE"],
            "impl": {"op": "update", "entity": "project", "id_param": 0,
                     "set_from_params": {"owner": 1}},
        },
        {
            "name": "archive_project",
            "desc": "Archive a project (kept, but read-only).",
            "params": [{"name": "project", "type": "ID:project",
                        "desc": "project to archive",
                        "field": ["project", "id"]}],
            "returns": "OBJ:project",
            "effects": ["WRITE"],
            "impl": {"op": "update", "entity": "project", "id_param": 0,
                     "set_const": {"status": "archived"}},
        },
        {
            "name": "delete_project",
            "desc": "Permanently delete a project and all its data.",
            "params": [{"name": "project", "type": "ID:project",
                        "desc": "project to delete",
                        "field": ["project", "id"]}],
            "returns": None,
            "effects": ["DELETE"],
            "impl": {"op": "delete", "entity": "project", "id_param": 0},
        },
        {
            "name": "notify_member",
            "desc": "Send a notification to a workspace member.",
            "params": [
                {"name": "user", "type": "ID:user", "desc": "recipient",
                 "field": ["user", "id"]},
                {"name": "text", "type": "STR", "desc": "notification text"},
            ],
            "returns": None,
            "effects": ["SEND"],
            "impl": {"op": "send", "channel": "notification",
                     "param_map": ["to", "text"]},
        },
    ],
    "default_state": {
        "entities": {
            "user": [
                {"id": "user_1", "name": "Sam Ortiz", "email": "sam@studio.test",
                 "manager": "user_3", "active": False},
                {"id": "user_2", "name": "Lena Fisher", "email": "lena@studio.test",
                 "manager": "user_3", "active": True},
                {"id": "user_3", "name": "Noor Hadid", "email": "noor@studio.test",
                 "manager": "user_3", "active": True},
                {"id": "user_4", "name": "Ty Walker", "email": "ty@studio.test",
                 "manager": "user_2", "active": False},
            ],
            "project": [
                {"id": "project_1", "name": "Website refresh", "owner": "user_1",
                 "status": "active", "last_activity": NOW - 200 * DAY},
                {"id": "project_2", "name": "Mobile app", "owner": "user_2",
                 "status": "active", "last_activity": NOW - 3 * DAY},
                {"id": "project_3", "name": "Data warehouse", "owner": "user_4",
                 "status": "active", "last_activity": NOW - 90 * DAY},
                {"id": "project_4", "name": "Legacy CMS", "owner": "user_1",
                 "status": "archived", "last_activity": NOW - 400 * DAY},
            ],
        },
        "outbox": [],
        "payments": [],
    },
}
