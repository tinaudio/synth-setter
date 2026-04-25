"""No-op fixture: ignores its CLI args and produces no stdout.

Used to verify that ``_resolve_pair`` rejects empty captured YAML — the shim
will still append ``--cfg job --resolve`` and redirect stdout to a file, but
because this script never prints anything, the file ends up empty (and
``yaml.safe_load`` returns ``None``).
"""
